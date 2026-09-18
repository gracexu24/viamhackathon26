# Viam Ball Grabber — Additive MVP Scope

## Objective

Pick up one stationary red stress ball using existing Viam resources:

`cam` → `color_detect` → `object-segmenter` → Viam frame system → `builtin` motion → `gripper`.

The first deliverable is a verified 3D ball position. A physical grab follows only after coordinate and gripper alignment checks in the real cell.

## Preserve Existing Code

Do **not** rename, move, overwrite, or delete any existing file or function. In particular preserve:

- `connection.py`
- `ball_tracking.py`
- `vision/live_ball.py`
- `vision/viam_pipeline.py`
- `motion/ball_pickup.py`
- `motion/default1.py`, `motion/default2.py`, and `motion/trajectory.py`

The local OpenCV detector and trajectory code are experiments. They are not used by this Viam-native MVP, but remain available for comparison later.

## Allowed Changes

Only additive changes are in scope:

1. Improve `vision/viam_pipeline.py` only to report existing Viam detector box midpoints and object-segmenter geometry.
2. Add `vision/viam_ball.py` after the segmenter works. It may provide `detect_ball(machine, ...)` and `locate_ball_3d(machine, ...)`.
3. Add `motion/viam_stationary_grab.py` after XYZ and gripper alignment are verified. It may provide one explicit `run_stationary_grab(...)` function.
4. Add targeted tests for new functions and concise README notes.

No repository-wide refactor, shared `types.py`, replacement `arm.py`, replacement `poses.py`, or edits to legacy motion files are in scope.

## P0 — Repair Viam 3D Localization

Current evidence:

- `color_detect` is healthy and detected `ball` at midpoint `(239.5, 402.0)`.
- `object-segmenter` returned `failed to get properties from source camera`.

Fix this in the Viam app before adding pickup code:

1. Configure `object-segmenter` to use healthy camera resource `cam`, never failed `cam2`.
2. Confirm `cam` has `color` and `depth` sensors with `align_color_depth: true`.
3. Restart/reconfigure the segmenter, then run:

   ```powershell
   .venv/Scripts/python.exe -m vision.viam_pipeline --camera cam --detector color_detect
   ```

P0 passes only when `object-segmenter` returns one geometry center and its reference frame. Do not derive depth from a 2D box for this Viam-native path.

## P1 — Verify XYZ Without Motion

Read and print the selected detection label, confidence, box midpoint, segment-center XYZ, and its Viam reference frame. Transform its center to `world` with `machine.transform_pose(...)` while the wrist-mounted camera is still.

Place the ball at known table positions. P1 passes when its world XYZ is repeatable across observe poses and within gripper tolerance. No movement is enabled.

## P2 — Approach Only

Add this new function without changing `motion/ball_pickup.py`:

```python
async def move_to_ball_approach(machine, ball_in_world, config) -> None:
    ...
```

Use `MotionClient.from_robot(machine, "builtin")` and move the configured `gripper` TCP to a `PoseInFrame(reference_frame="world", ...)`. Use an explicit approach pose 100 mm above the ball in world Z-up. Store the validated orientation and bounds in a small new config file.

P2 ends at the approach pose. It must not open, descend, grab, or lift.

## Measured TCP Baseline and Module Contract

Read-only Viam data collected on 2026-09-18:

```text
Current gripper TCP in world (mm)
X = 228.02, Y = -821.87, Z = 78.57

Current gripper orientation
o_x = -0.0770, o_y = -0.1226, o_z = -0.9895, theta = -129.94 degrees

Last detected ball center in world (mm)
X = 197.61, Y = -818.85, Z = 17.93
```

The gripper tool points almost straight down (`o_z ≈ -1`). The observed height
difference from those two samples is approximately `ball_z - gripper_z = -60.64 mm`.
This is a **candidate** TCP-to-ball grasp-height offset only: the samples were not
captured in one synchronized check, so it must be confirmed with the ball visible:

```powershell
.venv/Scripts/python.exe -m vision.viam_ball --include-gripper-pose
```

If the jaws are physically at the correct ball-grip height at that time, record:

```text
tcp_to_ball_z_mm = gripper_world_z - ball_world_z
```

Implementation update: the user explicitly designated the measured gripper
level as the grip height for this stationary table task. The added function uses
`grasp_z_world_mm = 78.57077725670194` as a taught height, with the measured
orientation. It checks detected ball Z against the original table level and
rejects height changes. The 60.64 mm historical difference is not a calibrated
offset and is not used by the implementation. A future calibrated offset mode
could instead compute:

```text
grasp_tcp = (ball_world_x + tcp_to_ball_x,
             ball_world_y + tcp_to_ball_y,
             ball_world_z + tcp_to_ball_z)
approach_tcp = grasp_tcp + (0, 0, approach_height_mm)
lift_tcp = grasp_tcp + (0, 0, lift_height_mm)
```

Set `tcp_to_ball_x` and `tcp_to_ball_y` to zero only after confirming that the
configured gripper TCP is centered between the jaws. Otherwise measure and store
their offsets too.

### Future `ball-grabber` Generic Service (not yet deployed)

The module should accept only a compact configuration:

```json
{
  "camera": "cam",
  "detector": "color_detect",
  "segmenter": "object-segmenter",
  "gripper": "gripper",
  "motion": "builtin",
  "world_frame": "world",
  "ball_label": "ball",
  "grasp_z_world_mm": 78.57077725670194,
  "expected_ball_z_world_mm": 17.933363077496036,
  "ball_height_tolerance_mm": 15,
  "grasp_xy_offset_mm": [0, 0],
  "approach_height_mm": 100,
  "lift_height_mm": 125,
  "grasp_orientation": [-0.0770, -0.1226, -0.9895, -129.94],
  "max_reacquire_shift_mm": 10
}
```

`grasp_z_world_mm` and `grasp_orientation` are recorded taught-pose values for
this table setup. They do not establish camera/hand-eye calibration. The actual
SDK entry point and complete configuration are `motion/viam_stationary_grab.py`
and `stationary_grab.config.json`. Preview is the default; `--execute` runs motion.

It exposes only:

```json
{"action": "detect_ball"}
{"action": "approach_ball"}
{"action": "grab_ball"}
```

`approach_ball` is the P2 safety command. It stops at `approach_tcp` and never
opens or closes the gripper. `grab_ball` runs this finite sequence:

```text
detect + localize while arm is still
→ transform ball to world
→ calculate grasp_tcp and approach_tcp
→ validate bounds and motion plan
→ move gripper to approach_tcp
→ re-detect + transform fresh camera result to world
→ abort if target moved more than max_reacquire_shift_mm
→ open gripper
→ move gripper to grasp_tcp
→ grab and verify holding
→ move gripper to lift_tcp
```

Every move uses the Viam `builtin` Motion service with `component_name="gripper"`
and a `PoseInFrame` in `world`. The service receives its Viam dependencies; it
does not create a `RobotClient`, call xArm directly, calculate IK, or modify
machine configuration.

## P3 — Controlled Stationary Grab

Only after P2 physical verification:

1. Move to the world-frame approach pose.
2. Open the gripper.
3. Move to the verified world-frame grasp pose.
4. Call `gripper.grab()` and require a true result.
5. Move to a world-frame lift pose 100–150 mm above grasp.

Transform every camera observation to `world` before planning. Never use an old wrist-camera-frame point after an arm move. On a failed plan, failed grasp, cancellation, or exception, stop arm and gripper and return a clear failure reason.

## P4 — Optional Module Packaging

Do not create a custom Viam module until P3 works from the local SDK command. Then package the proven grab function as a small Generic Service exposing only:

```json
{"action": "detect_ball"}
{"action": "grab_ball"}
```

It receives `cam`, `color_detect`, `object-segmenter`, `gripper`, and `builtin` motion as dependencies. It must not create another `RobotClient` or duplicate vision, depth, IK, or hardware-driver code.

## Out of Scope

- Refactoring the repository structure.
- Changing existing public functions.
- Custom OpenCV detection or RealSense depth projection for this Viam path.
- Direct xArm SDK calls or manual inverse kinematics.
- Config changes from code.
- Thrown-ball prediction, catching, and continuous visual servoing.
