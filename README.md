# viamhackathon26
Fine Motor Skills Hackathon

## Stationary grab using Viam services

New additive entry point: `motion/viam_stationary_grab.py`. It uses `cam`,
`color_detect`, `object-segmenter`, and Viam `builtin` motion to move `gripper`.
Existing preview, trajectory and pickup functions are preserved.

From the project directory:

```powershell
# Print fresh ball XYZ and the proposed gripper targets; no motion:
python -m motion.viam_stationary_grab
# Physical approach only (no gripper open/close):
python -m motion.viam_stationary_grab --execute --approach-only
# Physical stationary pickup, then lift:
python -m motion.viam_stationary_grab --execute
```

The default [stationary_grab.config.json](stationary_grab.config.json) uses the
user-designated grasp level measured at world Z **78.5708 mm**, preserving that
measured downward orientation. Detected world X/Y determines the new position.
It assumes the configured gripper origin is centered over the jaws in X/Y; adjust
`grasp_xy_offset_mm` for a measured horizontal offset. Ball Z is checked against
the observed table level (17.9334 mm, tolerance 15 mm). It is **not** used to derive
a tool offset from the old, separately captured measurements. These settings apply
only to this table, ball size and tool orientation.

The configured workspace is a deliberately small target box around those
readings: X 100–350, Y -950 to -650, Z 60–260 mm. These are application limits,
not measured arm reach or collision guarantees. Viam still plans against the
machine's configured kinematics and obstacle geometry. The grasp center/tool
frame and obstacles must be configured correctly in Viam. Change bounds only to
match the actual cell. No motion speed override is made: the arm's configured
speed remains in effect.

One cycle: take three stable 3D readings with the wrist stopped, transform to
world, raise to clearance if necessary, approach 100 mm above the taught grasp
level, reacquire the ball, align, open, recheck the ball, descend with a linear
constraint, grab, and lift 125 mm only if `grab()` returns true. It never uses
old camera coordinates after moving the wrist. Multiple ball segments, changed
height, unstable observations, failed moves and lost detections cause failure.
If a failure occurs after a motion command, arm/gripper stop requests are sent;
any failed stop requests are included in the output.

This is a stationary-object sequence, not continuous tracking during descent.
Use exclusive arm control. The ball must remain visible with usable depth from
the approach pose; loss of view aborts before descent. Segmenter responses have
no exposure timestamps, so wrist stillness checks and repeated reads do not
provide synchronized moving-camera tracking. No automatic home pose or retry is
issued. The Python function is `run_stationary_grab(machine, config,
execute=False, approach_only=False)`; registry module packaging remains separate.

## Wrist-camera red ball pickup

### Use the existing Viam vision services

This project already uses the official `viam-sdk` Python package and `RobotClient`.
To read the machine's existing services instead of running local color detection:

```powershell
.venv/Scripts/python.exe -m vision.viam_pipeline --camera cam --detector color_detect
```

This read-only command lists resources and vision capabilities, reads bounding
boxes and their midpoints from `color_detect`, and requests 3D geometry centers
from services supporting object point clouds. Results retain their reported
reference frame; they are not automatically world/arm coordinates. No calibration
commands, machine-config edits, or arm movements are issued. It uses `cam`, not
the unhealthy `cam2` shown in the app. The existing preview/pickup still uses the
local detector until the live services and their camera/frame mapping are verified.

For the scoped Viam-native P1 check, which selects the highest-confidence `ball`,
requires a matching 3D segment, then reports its camera and world coordinates:

```powershell
.venv/Scripts/python.exe -m vision.viam_ball --camera cam --detector color_detect --segmenter object-segmenter --label ball
```

This is read-only. It deliberately fails rather than estimating depth locally if
the segmenter is unhealthy, returns no `ball` label, or cannot transform the
camera-frame point to `world`.

To read the gripper TCP position even when no ball is visible:

```powershell
.venv/Scripts/python.exe -m vision.viam_ball --gripper-pose-only
```

MCP is a separate interface for an AI client; it is not required for SDK access.
No Viam MCP endpoint has been configured or connected in this project.

`ball_tracking.py` provides `detect_red_ball` (RGB + depth to XYZ), `BallCamera.observe`
(XYZ transformed into the Viam world frame), and `Trajectory.predict` (short-horizon
position/velocity prediction). `motion/ball_pickup.py` provides
`move_to_ball_and_grip(machine, config, execute=False)` and a runnable command.
These are Python functions using existing Viam components, not a registry module.

The initial task is a **stationary ball** with a **wrist-mounted D435i**. The arm
stops while observing, approaches above the ball, observes again, moves the grasp
center to the ball, verifies its location, and calls `gripper.grab()`. No automatic
lift follows. Observation is the default; `--execute` enables physical movement.

### Configure before running

1. Keep the existing `.env` variables `VIAM_ADDRESS`, `VIAM_API_KEY_ID`, and
   `VIAM_API_KEY`. Never commit credentials.
2. In Viam, confirm the actual UFactory arm model from its label/configuration.
   The Python code uses your `arm`, `cam`, and `gripper` component names and does
   not assume that “x115” identifies a supported arm model.
3. Configure the RealSense with `sensors: ["color", "depth"]` and
   `align_color_depth: true`. Confirm the actual source names returned by
   `cam.get_images()` and set `color_source` / `depth_source` accordingly. Color
   must be JPEG/PNG; depth must be Viam raw depth (`image/vnd.viam.dep`, millimeters).
4. Calibrate the rigid camera-to-wrist transform (hand-eye calibration) and enter
   it in Viam's frame tree as a child of the arm end-effector. Set
   `camera_optical_frame` to the frame whose axes are **X right, Y down, Z forward**
   from the color camera. A camera housing frame may require an extra fixed
   optical-frame transform. The code uses Viam `transform_pose` with current joints;
   a single fixed camera-to-world matrix is wrong for a wrist-mounted camera.
5. Configure the gripper as a child of the arm with its frame origin at the actual
   grasp center, including the tool offset. Configure table, walls and other
   collision geometry in Viam's frame system. Workspace bounds below only check
   target positions; they do not replace collision geometry or joint limits.
6. Copy `ball_config.example.json` to `ball_config.json`. **All numeric camera,
   orientation and workspace values in the example are placeholders.** Enter the
   aligned **color** stream's calibrated intrinsics at its actual resolution and
   measured ball radius. Use rectified images; distortion is not corrected here.
   Do not substitute depth-stream intrinsics for color-stream intrinsics.
7. Set a Z-up world frame, measured reachable `workspace_mm`, and a verified
   downward `grasp_orientation` using Viam orientation-vector notation (degrees).
   Validate the XYZ of a stationary ball in several poses of the wrist: it should
   remain fixed in world coordinates. Then set `calibration_verified` to true.

### Run

For a live, read-only RGB/depth window with a red-ball box, box midpoint, camera
XYZ and pixel alignment error:

```powershell
.venv/Scripts/python.exe -m vision.live_ball --camera cam --radius-mm 20
# Or use your measured settings:
.venv/Scripts/python.exe -m vision.live_ball --config ball_config.json
```

Without a config it queries intrinsics from the camera; verify those describe the
aligned color stream. The default ball radius is 20 mm; change it to your measured
radius. The yellow midpoint samples surface depth and deprojects the ball center.
The cyan target defaults to the optical principal point; optionally set
`alignment_pixel: [u, v]` in your config. The pixel error is ball midpoint minus
target (right/down positive). This is visual guidance, not automatic arm alignment;
the camera-to-gripper offset still needs calibration. Press Escape to close.

From this directory, using Python 3.10+:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
Copy-Item ball_config.example.json ball_config.json
# Edit ball_config.json with measured calibration first.
.venv/Scripts/python.exe -m motion.ball_pickup --config ball_config.json
# Once observation, frames and workspace have been validated:
.venv/Scripts/python.exe -m motion.ball_pickup --config ball_config.json --execute
```

The first command reports predicted world XYZ in millimeters, velocity in mm/s,
and fit error without actuating anything. Camera and client clocks must be
synchronized; frames older than 250 ms are rejected by default. Use exclusive
control of the arm during this process. Captures occur after settling and joint
positions are checked before/after acquisition; this is not exposure-synchronized
pose tracking for a moving camera.

The ball must remain visible with valid depth even at the final grasp pose. If
the wrist mount occludes it or brings it inside the camera's usable depth range,
pickup aborts before closing. Change the mount/viewing geometry for that case.
Red circular distractors, ambiguous detections, missing depth, stale frames,
moving targets and failed plans also abort. Camera depth plus ball radius gives
an approximate sphere center; verify its error against your gripper tolerance.

### Later: thrown balls

`Trajectory(acceleration=(0, 0, -9810))` predicts free flight in a Z-up world;
zero acceleration models short constant-velocity segments. Add timestamped world
observations, then call `predict(future_unix_time)` to get position, velocity and
fit residual. Predictions are limited to 0.5 s and do not model bounces or drag.
The pickup routine rejects fast balls and is **not a thrown-ball catcher**.
Catching needs measured arm/gripper latency, feasible interception planning,
continuous tracking with timestamp-synchronized camera poses (or a fixed camera),
and a suitable real-time control loop. Its lead-time setting alone does not
synchronize arm arrival with a flying ball.

### Verification and references

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Tests use synthetic images/trajectories and mocked motion; they do not move hardware.

- [Viam RealSense alignment setup](https://docs.viam.com/tutorials/pick-and-place/configure-resources/)
- [Viam frame-system API](https://docs.viam.com/motion-planning/reference/frame-system-api/)
- [Viam perception-guided picking](https://docs.viam.com/tutorials/pick-and-place/perception-guided-picking/)

## Rolling trajectory tracker

From the project directory on Linux, with the existing `.env` credentials:

```bash
.venv/bin/python -m motion.trajectory_rolling --hz 60
```

Uses `ball_catch.config.json` by default; override with `--config PATH`.
Keep the wrist stationary. Ctrl+C stops tracking. This command reports world
position in meters, velocity in m/s, and upward threshold crossings; it does not
command the arm. `--print-hz 5` controls console output independently of sampling.

The default localization target is 60 Hz (16.67 ms), independent of the catch
script's 20 Hz setting. Processing time counts toward that budget. The displayed
localization rate measures successful results, not camera FPS. Segmenter results
lack capture timestamps, so velocity uses completion times and is approximate.

Actual 60 FPS capture requires a camera driver and stream profile supporting it.
The [Viam RealSense module's documented attributes](https://github.com/viam-modules/viam-camera-realsense#attributes)
do not expose an FPS option. This script cannot set or verify the sensor rate;
confirm the installed driver and supported color/depth profiles before assuming
60 FPS capture. Network calls and segmentation can reduce localization throughput
even when the camera captures at 60 FPS.

## Tracker installed on armfarm14

The local image tracker is installed in `/opt/viam/trajectory-local` on part
`49d63d4f-4191-4d94-9e43-e379f846dd0e`, with its own Python environment at
`/opt/viam/trajectory-local-venv`. Run from your laptop:

```bash
viam machine part shell --part 49d63d4f-4191-4d94-9e43-e379f846dd0e
```

Then inside that shell:

```bash
/opt/viam/trajectory-local/run.sh
```

Ctrl+C stops it. `--duration 10` runs a bounded check; `--print-hz 2` reduces output.
The source is `motion/trajectory_local.py`. It connects to Viam over loopback TLS,
reads existing machine credentials in memory, and detects the red ball locally.
It does not run the object segmenter or command hardware. The launcher does not
contain API secrets. No service/autostart is installed.

**Default output is pixel UV and pixel velocity, not world XYZ.** Alignment of
this custom module's returned depth and color has not been verified. Only after
verifying alignment and color intrinsics, use `--aligned-depth --radius-mm 20`
(substitute the measured ball radius) for camera-optical XYZ in meters and
velocity in m/s. Camera Z points forward, not upward; this mode does not report
world-frame throw detection. Missing/ambiguous detections reset the velocity
history. The default radius is an assumption, not a measured calibration.

The installed camera config contains `fps: 60`. A local three-second probe on
2026-09-18 measured 59.53 distinct frame timestamps/s, while segmentation took
234 ms per call. This custom build may differ from the stock module documentation
above. The tracker polls at up to 120 Hz to consume that 60 FPS stream, skips
repeated/stale timestamps, and reports fresh-frame and successful-detection rates
separately. It does not reconfigure the camera's capture rate.
