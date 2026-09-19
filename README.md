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

## Side-profile camera (`cam2`) red-ball tracking

The side camera is an existing Viam resource. Check its color stream from the
project directory, without moving the arm:

```powershell
.venv/Scripts/python.exe -m vision.live_ball --camera cam2 --probe
.venv/Scripts/python.exe -m vision.side_tracker --camera cam2 --duration 10
# Optional annotated preview on the laptop (Escape or q closes it):
.venv/Scripts/python.exe -m vision.side_tracker --camera cam2 --display
```

The laptop/cloud path is for framing and diagnostics, not fast flight tracking:
one check on 2026-09-18 delivered only 0.2–0.9 fresh frames/s with 1.5–2.6 s
capture-to-receipt age. The tracker rejects frames older than 0.5 s by default.
For real-time pixel tracking, use the existing on-robot install:

```bash
viam machines part shell --part 49d63d4f-4191-4d94-9e43-e379f846dd0e
cd /opt/viam/trajectory-local
./run.sh --camera cam2 --duration 10 --print-hz 3
```

The local `cam2` test returned about 59–60 fresh frames/s with no stale frames.
It reports ball pixel position and velocity only when the red ball appears as
one distinct circular contour. A hand touching/overlapping the ball can merge
the red mask with skin and cause `No single red ball`; first test with the ball
alone in view. No command above moves the arm or uses the 3D segmenter.

### Record and fit a side-camera flight

Run the flight tracker on the compute device so it receives the camera's 60 FPS
stream rather than delayed cloud frames:

```bash
cd /opt/viam/trajectory-local
/opt/viam/trajectory-local-venv/bin/python -m motion.side_flight_local \
  --machine-config /root/.viam/cached_cloud_config_49d63d4f-4191-4d94-9e43-e379f846dd0e.json \
  --camera cam2 --duration 15 --record /tmp/cam2-flight.jsonl
```

After measuring the image column at which the basket catches the ball, add for
example `--catch-u-px 700`. The output fits horizontal image motion linearly and
vertical image motion quadratically using camera capture timestamps, then reports
the predicted crossing time and image height. It remains read-only.

Because `cam2` is angled and perspective-projected, its pixel parabola is an
image-space approximation for a constrained throw lane. Pixel acceleration is
not gravity and pixels must not be passed to the arm as millimeters. The physical
model in `motion/trajectory_fit.py` uses
`p(t) = p0 + v0*t + 0.5*[0,0,-9810]*t^2`, but only accepts calibrated, Z-up world
XYZ samples. Supplying those requires camera-to-world calibration plus depth or
another independent view; the current side-camera tracker intentionally does not
invent the missing depth.

For throws constrained to one vertical plane, full 3D is unnecessary. Measure at
least four non-collinear points in that physical plane, record their matching
`cam2` pixels, and replace every placeholder in
`side_camera_plane.example.json`. The mapping removes the camera's sideways
angle and perspective within that plane. Copy the measured file to the compute
device and run:

```bash
/opt/viam/trajectory-local-venv/bin/python -m motion.side_flight_local \
  --machine-config /root/.viam/cached_cloud_config_49d63d4f-4191-4d94-9e43-e379f846dd0e.json \
  --camera cam2 --plane-calibration /path/to/measured-plane.json \
  --catch-x-mm 1800 --duration 15 --record /tmp/cam2-flight.jsonl
```

This reports physical plane `(X,Z)`, velocity, fit error, and predicted time and
height at the catch line using `Z(t)=Z0+Vz*t-0.5*9810*t^2`. It is valid only while
the ball stays close to the calibrated plane; substantial toward/away motion
requires depth or a second calibrated view.

### Read-only 3D ball-to-arm guidance logic

`vision/relative_3d.py` contains perception and decision logic only; it imports no
arm or motion client and cannot command hardware. From one aligned `cam2` frame it:

1. detects the red ball and uses its depth plus measured radius to estimate the
   ball center in camera XYZ;
2. takes the robust median depth of a tracked arm/basket reference ROI and
   deprojects its center into the same XYZ frame;
3. calculates `ball_xyz - arm_reference_xyz`; and
4. reports left/right, up/down, and toward/away recommendations outside configured
   millimeter deadbands.

Camera optical axes are X right, Y down, and Z away from the camera. Because the
side camera is angled, those labels are camera-relative. Passing a calibrated
camera-to-world rotation converts the difference to world-axis directions; its
translation is unnecessary for a same-frame difference. The configured ROI must
cover a visible solid reference or marker on the arm/basket. An ROI over the empty
basket opening measures the background and is invalid. If that marker is offset
from the true basket center, measure and account for that offset before any future
motion layer uses the result.

The output is not itself a safe motion command: it has no arm reachability,
collision, latency, or trajectory checks. For a catch, compare the arm reference
against the ball position predicted at the catch time—not merely the ball's latest
position.

## Read-only two-camera yellow-ball predictor

`motion/two_camera_rgb_local.py` is the current RGB-only catch predictor. It does
not import or call the arm, motion service, gripper, depth, point clouds, or a
vision segmenter. Run it on the Viam compute machine so both camera loops receive
local capture-timestamped frames:

```bash
cd /opt/viam/trajectory-local
/opt/viam/trajectory-local-venv/bin/python -m motion.two_camera_rgb_local \
  --machine-config /root/.viam/cached_cloud_config_49d63d4f-4191-4d94-9e43-e379f846dd0e.json \
  --side-camera cam2 --front-camera cam \
  --catch-u-px "$CATCH_U_PX" \
  --basket-u-px "$BASKET_U_PX" --basket-v-px "$BASKET_V_PX" \
  --duration 20
```

Set `CATCH_U_PX` to the desired measured side-image catch column, and set
`BASKET_U_PX` and `BASKET_V_PX` to the measured front-image basket center. Use
`--front-camera cam1` if that is the actual front resource. The
side fit estimates the future crossing timestamp; the front fit predicts ball
U and V at that exact timestamp and returns a structured `CatchPrediction` with
both signed errors. `--invert-lateral` only affects the legacy LEFT/RIGHT label;
Phase 3 uses its own calibrated axis signs.

Default yellow HSV thresholds use OpenCV ranges: H 18–40, S at least 90, and V at
least 80. The detector returns multiple candidates; a constant-velocity pixel
tracker associates the nearest plausible candidate and survives a 0.1 s gap
without inserting fake observations.

The side fit accepts either image direction. It emits a prediction only when the
fitted crossing is in the future and within `--max-horizon-s`; a ball moving away
from `catch_u_px` is therefore rejected.

## Phase 3: one-shot 2D basket movement in the catch plane

`motion/intercept_controller.py` consumes one structured Phase 2 prediction and
maps front-image horizontal and vertical error onto two configured UF850
Cartesian axes. The third Cartesian coordinate is forced to the configured
fixed catch-plane value, and basket orientation is copied unchanged from the
verified catch-ready pose. Perception contains no arm calls.

Copy `intercept_controller.config.example.json` to
`intercept_controller.config.json` and replace every `null` with a physically
measured value. The example is deliberately non-runnable: the actual axes,
directions, scales, pose, bounds, and lead time must not be guessed.

Integrated live dry run (the default):

```bash
cd /opt/viam/trajectory-local
/opt/viam/trajectory-local-venv/bin/python -m motion.intercept_controller \
  --config intercept_controller.config.json \
  --machine-config /root/.viam/cached_cloud_config_49d63d4f-4191-4d94-9e43-e379f846dd0e.json \
  --arm arm --side-camera cam2 --front-camera cam1 \
  --catch-u-px "$CATCH_U_PX" \
  --basket-u-px "$BASKET_U_PX" --basket-v-px "$BASKET_V_PX"
```

Add `--execute` only after dry-run and no-ball motion calibration. A live run
commits the first sufficiently valid prediction, issues at most one
`move_to_position`, arrives early, and holds. Restart the command for the next
attempt. A saved synthetic/Phase 2 result can be tested without cameras using
`--prediction-json phase2_prediction.json`.

The controller validates prediction age and timing, both fit errors, independent
deadbands, correction limits, rectangular in-plane bounds, fixed plane normal,
fixed orientation, and arm idle state. A correction requiring clamping is shown
by the pure mapping function but is rejected before live motion. Dry run never
gets an arm resource. A failed movement is stopped and never retried.

Calibration order: teach the catch-ready pose; jog each candidate Cartesian axis
by a small known distance; identify image horizontal, image vertical, and the
remaining plane-normal axis; measure direction and pixels traveled independently;
then establish conservative in-plane bounds and minimum lead time. Test center,
left, right, up, down, diagonal, and center again without a ball before live use.
