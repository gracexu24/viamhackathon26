# Viam Two-Camera Basket Catch — Scoped Implementation Plan

## Objective

Catch a thrown red ball in mid-air using:

- **Side-profile camera** to observe the ball's flight and estimate where/when it will reach the robot.
- **Wrist camera** to provide additional ball position information as the ball approaches the robot.
- **UFACTORY 850 + basket** as the catching mechanism.
- **Viam** for cameras, arm control, and robot resources.

The goal is **not** to optimize Viam, build a custom low-latency driver, use ROS, or create a sophisticated visual-servo controller.

The goal is to collect enough useful information from both cameras to predict a catch position and command the basket there in time.

---

# Core System

```text
                   SIDE CAMERA
                        |
                        v
                red ball (u, v, t)
                        |
                        v
             estimate flight trajectory
                        |
                        v
               predicted catch area
                 (height + timing)
                        |
                        |
                        +----------------------+
                                               |
                                               v
                                      CATCH CONTROLLER
                                               |
                                               |
                   WRIST CAMERA                |
                        |                      |
                        v                      |
                red ball (u, v)                |
                        |                      |
                        v                      |
              refine basket target ------------+
                                               |
                                               v
                                         VIAM ARM
                                               |
                                               v
                                         UFACTORY 850
                                               |
                                               v
                                            BASKET
```

---

# Scope Decisions

## We are doing

- Detecting one clearly colored ball.
- Tracking it from a fixed side camera.
- Tracking it from the wrist camera when visible.
- Using timestamps from camera frames.
- Estimating a simple projectile path.
- Predicting an interception point.
- Moving a basket to that interception point.
- Using the wrist camera as extra information to improve the target before the catch.
- Keeping the basket open at all times.

## We are not doing

- ROS.
- Gripper open/close timing.
- GGCNN.
- YOLO unless HSV fails badly.
- Reinforcement learning.
- Custom Viam arm drivers.
- Modifying the Viam UFACTORY module.
- High-rate custom servo control.
- Perfect stereo reconstruction.
- Full 3D SLAM.
- Complex MPC.
- Catching arbitrary throws from every direction.

The first working version should assume the throw comes from a **known general direction toward a known catch region**.

---

# Existing Repo

Repository:

`gracexu24/viamhackathon26`

Keep the existing stationary-ball and previous catch experiments intact.

Important existing pieces that can be reused:

```text
connection.py
vision/live_ball.py
vision/viam_ball.py
vision/viam_pipeline.py
motion/default1.py
motion/default2.py
motion/trajectory_local.py
motion/viam_ball_catch.py
ball_tracking.py
```

Do not spend time rewriting the existing stationary-grab system.

The new basket catcher should be a separate path.

---

# Proposed New Structure

```text
vision/
    side_tracker.py
    wrist_tracker.py
    trajectory.py

motion/
    basket_positions.py
    basket_catch.py

basket_catch.py
basket_catch.config.json
```

Keep this small.

---

# Phase 1 — Side Camera 2D Tracking

## Goal

Get reliable `(u, v, timestamp)` observations of the red ball from the side camera during a real throw.

This is the first thing that needs to work.

## File

```text
vision/side_tracker.py
```

## Input

Viam camera resource:

```text
side_cam
```

The exact name can be changed in config.

Use the Viam camera API directly:

```python
images, metadata = await camera.get_images(...)
```

Use the camera frame capture timestamp from `metadata`.

## Detection

Reuse the simple HSV red-ball logic already used in:

```text
motion/trajectory_local.py
ball_tracking.py
```

Do not use the Viam object segmenter for this path.

Do not use depth initially.

For each frame return:

```python
SideObservation(
    timestamp=...,
    u=...,
    v=...,
    radius_px=...
)
```

## Display

Draw:

- ball center
- bounding circle
- recent trajectory trail
- frame timestamp
- current pixel velocity

Example:

```text
SIDE CAMERA

+--------------------------------------+
|                                      |
|       o                              |
|          o                           |
|             o                        |
|                O  <- current ball    |
|                                      |
|                         |            |
|                         | catch zone |
|                         |            |
+--------------------------------------+
```

## Pass Condition

Phase 1 passes when:

- the ball is detected for most of a normal throw,
- fresh frames arrive consistently,
- timestamps increase correctly,
- the trail visually follows the ball,
- the tracker outputs several observations before the ball reaches the robot.

Nothing moves yet.

---

# Phase 2 — Wrist Camera Tracking

## Goal

Get simultaneous or near-simultaneous red-ball observations from the wrist camera.

The wrist camera is **not responsible for the full trajectory**.

Its purpose is to provide information about where the ball is relative to the basket as it gets closer.

## File

```text
vision/wrist_tracker.py
```

## Input

Existing wrist Viam camera:

```text
cam
```

## Detection

Use the same lightweight red HSV detector.

Initially return:

```python
WristObservation(
    timestamp=...,
    u=...,
    v=...,
    radius_px=...
)
```

No object segmenter.

No depth requirement.

No world transform requirement.

## Important Physical Check

Before writing control code, physically verify that the wrist camera can actually see the incoming ball when:

- the basket is mounted,
- the arm is in its expected catch posture,
- the ball approaches from the intended throw direction.

If the basket blocks the wrist camera, move or angle the camera.

## Pass Condition

Phase 2 passes when the wrist camera reliably sees the ball for at least part of its final approach.

---

# Phase 3 — Side-Camera Trajectory Estimate

## Goal

Estimate where and when the ball will arrive near the robot.

## File

```text
vision/trajectory.py
```

## Keep the model simple

From the side camera collect:

```text
(t0, u0, v0)
(t1, u1, v1)
(t2, u2, v2)
...
```

Fit:

```text
u(t) ≈ linear
v(t) ≈ quadratic
```

Conceptually:

```text
u(t) = a*t + b

v(t) = c*t^2 + d*t + e
```

Do not add a Kalman filter unless noisy measurements make the simple fit unusable.

Use roughly the latest:

```text
5–10 observations
```

from the throw.

---

# Phase 4 — Define a Catch Region

The basket does not need to chase the ball through arbitrary 3D space.

Choose a physical region where catches will happen.

Conceptually:

```text
SIDE VIEW


ball
   o
      o
         o
            o
               o

                    +---------+
                    | CATCH   |
                    | REGION  |
                    +---------+

                       basket
```

For the first demo, throws should consistently pass through this region.

The side camera needs to determine:

```text
when will the ball enter the catch region?
how high will it be?
```

The wrist camera helps determine:

```text
is the ball left/right of the basket?
is our initial target visibly wrong?
```

---

# Phase 5 — Calibrate Side Pixels to Basket Positions

Do not build a full 3D camera calibration unless necessary.

Instead, move the basket manually to several safe catch positions and record:

```text
side-camera ball/catch pixel height
UF850 joint position
basket physical height
```

Example:

```text
Catch position A
side v = 410 px
height ≈ 650 mm
joints = [...]

Catch position B
side v = 350 px
height ≈ 750 mm
joints = [...]

Catch position C
side v = 290 px
height ≈ 850 mm
joints = [...]

Catch position D
side v = 230 px
height ≈ 950 mm
joints = [...]
```

## File

```text
motion/basket_positions.py
```

Store a small set of taught basket poses.

Example:

```python
BASKET_POSES = [
    {
        "height_mm": 650,
        "side_v": 410,
        "joints": [...]
    },
    ...
]
```

Initially choose the closest taught pose.

Interpolation can be added after the discrete version works.

This avoids needing perfect camera-to-world calibration for the first catch.

---

# Phase 6 — First Moving Basket Test

## Goal

Use only the side camera prediction to move the basket.

No wrist correction yet.

Pipeline:

```text
side camera
    |
    v
collect 5–10 ball points
    |
    v
fit trajectory
    |
    v
predict catch height + time
    |
    v
select nearest taught basket pose
    |
    v
command UF850
    |
    v
hold basket there
```

## File

```text
motion/basket_catch.py
```

Do not try to continuously chase the ball.

Once the trajectory prediction becomes stable enough:

1. Choose the basket target.
2. Send the arm there.
3. Hold.
4. Let the ball arrive.

A large basket provides the tolerance.

## Pass Condition

The basket moves to a reasonable predicted position before the ball arrives.

It does not need to catch yet.

---

# Phase 7 — Get First Actual Catch

At this stage:

- Keep the throw direction controlled.
- Use the same thrower.
- Use roughly the same starting point.
- Keep the arm in a consistent starting posture.
- Use a large/light basket.
- Keep the basket opening facing upward.

The system is:

```text
SIDE CAMERA
     |
     v
2D trajectory
     |
     v
predicted interception
     |
     v
basket target
     |
     v
VIAM ARM COMMAND
     |
     v
BASKET WAITS
     |
     v
BALL ENTERS BASKET
```

Before adding more code, attempt repeated physical catches and determine what the dominant error is:

```text
too high / too low
too early / too late
too far left / too far right
```

This tells us what the wrist camera actually needs to correct.

---

# Phase 8 — Add Wrist-Camera Correction

Only add this after the side-camera system can put the basket approximately in the correct place.

## Goal

Use the wrist camera to make one simple final correction.

Example wrist image:

```text
+--------------------------------+
|                                |
|          ball ●                |
|                                |
|               +                |
|         basket center          |
|                                |
+--------------------------------+
```

Calculate:

```python
du = ball_u - basket_center_u
dv = ball_v - basket_center_v
```

Initially, use the wrist camera primarily for **lateral correction**.

The side camera already provides:

```text
height
flight timing
```

The wrist camera provides:

```text
left/right error
possibly near/far visual error
```

Do not attempt a complicated continuous visual-servo loop initially.

Instead use the wrist observation to choose among:

```text
basket left
basket center
basket right
```

or make one small target adjustment.

That is enough for the MVP.

---

# Phase 9 — Combine Both Cameras

Final MVP decision logic:

```text
SIDE CAMERA
    |
    +--> predicted catch height
    |
    +--> predicted catch time
    |
    v
initial basket target
    |
    |
WRIST CAMERA
    |
    +--> ball left/right of basket
    |
    v
small correction
    |
    v
FINAL BASKET TARGET
    |
    v
UFACTORY 850
```

The cameras do not need to produce one perfect fused 3D state.

They can provide complementary information.

That is intentionally simpler.

---

# Main Orchestrator

## File

```text
basket_catch.py
```

Responsibilities:

```python
connect_to_viam()

start_side_tracker()
start_wrist_tracker()

wait_for_throw()

side_samples = collect_side_samples()

prediction = predict_trajectory(side_samples)

basket_target = choose_basket_pose(prediction)

wrist_observation = get_latest_wrist_ball()

basket_target = refine_target(
    basket_target,
    wrist_observation
)

move_basket(basket_target)

hold_until_ball_arrives()
```

Keep orchestration readable.

Avoid putting camera processing, trajectory math, and arm motion into one giant file.

---

# Suggested Config

## `basket_catch.config.json`

```json
{
  "side_camera": "side_cam",
  "wrist_camera": "cam",
  "arm": "arm",
  "ball_color": "red",

  "side_history_samples": 8,
  "min_side_samples": 5,

  "catch_region_u_px": 900,

  "wrist_center_u_px": 640,
  "wrist_center_v_px": 360,

  "wrist_left_threshold_px": -40,
  "wrist_right_threshold_px": 40,

  "max_tracking_time_s": 3.0
}
```

Exact numbers should be filled in from the real camera setup.

---

# Recommended Implementation Order

## P0 — Side tracker

Build:

```text
vision/side_tracker.py
```

Verify real thrown-ball `(u,v,t)` tracking.

---

## P1 — Wrist tracker

Build:

```text
vision/wrist_tracker.py
```

Verify the ball is visible during final approach.

---

## P2 — Trajectory predictor

Build:

```text
vision/trajectory.py
```

For recorded or live throws print:

```text
catch in: 0.41 s
predicted side-camera height: 302 px
```

No robot motion.

---

## P3 — Basket pose table

Build:

```text
motion/basket_positions.py
```

Teach several safe UF850 basket positions.

Verify each manually.

---

## P4 — Side-camera-only basket movement

Build:

```text
motion/basket_catch.py
```

Use prediction to select and command a basket position.

---

## P5 — First catches

Test repeated controlled throws.

Record miss direction.

Do not add more complexity until there is evidence showing what is wrong.

---

## P6 — Wrist correction

Use wrist-camera ball location to make one final target adjustment.

---

## P7 — Integrated demo

Run:

```powershell
python basket_catch.py
```

Desired behavior:

```text
Waiting for throw...

Ball detected.
Side samples: 8
Trajectory fit valid.

Predicted catch:
  time:   0.46 s
  height: 812 mm

Initial basket target:
  middle-high

Wrist ball:
  52 px left of basket center

Adjustment:
  move basket left

Moving arm...

CATCH
```

---

# What We Should Reuse From the Existing Repo

## Reuse directly

From:

```text
motion/trajectory_local.py
```

Reuse:

- Viam local camera access pattern.
- frame timestamp handling.
- red HSV detector.
- duplicate-frame handling.

From:

```text
connection.py
```

Reuse the existing Viam connection logic.

From:

```text
motion/default1.py
motion/default2.py
```

Reuse the idea of known safe arm poses while recording basket positions.

From:

```text
ball_tracking.py
```

Reuse useful red-ball contour filtering logic if needed.

---

# What We Should Not Reuse for the Main Catch Loop

Do not base the new basket catcher directly on:

```text
motion/viam_ball_catch.py
```

That implementation assumes:

- 3D object segmentation,
- wrist-camera world localization,
- stationary-camera assumptions,
- Cartesian Motion service interception,
- gripper use.

Those assumptions do not match the new two-camera basket approach.

Keep the file as an experiment/reference.

---

# MVP Definition

The hackathon MVP is complete when:

1. The side camera tracks a thrown red ball.
2. The system estimates a plausible interception position.
3. The wrist camera sees the ball during final approach.
4. The robot selects/moves to a basket target using both observations.
5. The robot catches the ball in the basket on controlled throws.

It does **not** need to:

- work from every throw angle,
- recover perfect 3D coordinates,
- continuously servo at high frequency,
- use ML,
- use a custom robot controller.

---

# Simplest Successful Version

If time becomes tight, reduce the system to:

```text
side camera
     |
predict one of 5 vertical catch zones
     |
wrist camera
     |
choose LEFT / CENTER / RIGHT
     |
     v
5 heights × 3 lateral positions
     |
15 pre-taught basket poses
     |
     v
move UF850
     |
     v
catch
```

This is a very strong fallback because it still uses:

- two-camera perception,
- trajectory prediction,
- Viam,
- robot motion,
- autonomous physical interception,

without requiring perfect continuous 3D control.

The system only has to classify the predicted interception into one of a small number of basket locations.

That should be preferred over adding sophisticated control code if the continuous target approach is unstable.
