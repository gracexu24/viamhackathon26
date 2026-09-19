# Two-Camera Yellow-Ball Predictor Scope

## Objective

Use two fixed RGB camera streams to predict where a clearly visible yellow ball
will cross a configured catch region:

- `cam2` supplies the side view and estimates the future catch timestamp.
- `cam` supplies the front view and estimates lateral position at that timestamp.
- The output is `LEFT`, `CENTER`, or `RIGHT` relative to the measured basket
  center pixel.

This phase is perception-only. It does not command an arm, gripper, or Motion
service.

## Implemented data flow

```text
cam2 color frames
  -> yellow HSV candidates
  -> temporal association
  -> u(t) linear fit and v(t) quadratic fit
  -> future crossing of catch_u_px
  -> catch timestamp

cam color frames
  -> yellow HSV candidates
  -> temporal association
  -> front u(t) linear fit
  -> evaluate at side catch timestamp
  -> compare with basket_u_px
  -> LEFT / CENTER / RIGHT
```

Both streams use camera capture timestamps. Duplicate, stale, ambiguous, and
poor-fit observations are rejected.

## Current modules

| Path | Responsibility |
| --- | --- |
| `motion/two_camera_rgb_local.py` | Concurrent two-camera predictor and CLI. |
| `motion/side_flight_local.py` | Side-camera-only flight diagnostics and optional plane fit. |
| `motion/trajectory_fit.py` | Pixel, throw-plane, and world trajectory fitting. |
| `vision/yellow_ball.py` | Yellow HSV detection and temporal association. |
| `vision/local_camera.py` | Read-only local Viam camera helpers. |
| `vision/transforms.py` | Validated rigid transforms for calibrated 3D points. |
| `cam2_transform.json` | Measured `T_world_from_cam2` and matching Viam frame block. |

## Coordinate frames

The RGB predictor is intentionally pixel-space. A pixel identifies a camera ray,
not an absolute 3D point, so `T_world_from_cam2` cannot be applied until depth or
a calibrated ray/plane intersection provides camera-frame XYZ.

For 3D work, use column vectors and millimetres:

```text
p_world = T_world_from_cam2 @ [x_cam2, y_cam2, z_cam2, 1]
```

`cam2` is fixed, so its Viam frame parent must be `world`. The wrist camera and
tool frames, if used in a future motion phase, must be defined through Viam's
arm/frame chain rather than copied from one arm pose.

## Required calibration

- Side catch-line column, `catch_u_px`.
- Front basket-center column, `basket_u_px`.
- Camera clock freshness and capture timestamps.
- Yellow HSV/circularity thresholds for the actual lighting.
- For metric side-plane fitting: measured image-to-plane correspondences.
- For absolute 3D: aligned depth or ray/plane intersection, plus verified
  `T_world_from_cam2`.

## Safety boundary

- The implemented predictor has no robot-motion imports or actuator calls.
- Pixel predictions are not arm targets.
- A future motion phase must add reachability, collision, timing, workspace,
  exclusive-control, stop-on-failure, and arrival verification before execution.
- Every camera-to-world calibration must be checked against multiple measured
  points in the physical workspace.

## Run

```bash
python -m motion.two_camera_rgb_local \
  --machine-config /path/to/cached-machine-config.json \
  --side-camera cam2 \
  --front-camera cam \
  --catch-u-px 700 \
  --basket-u-px 424 \
  --duration 20
```

Use `python -m motion.two_camera_rgb_local --help` for threshold and timing
options.

## Acceptance criteria

1. Both cameras deliver fresh frames with valid capture timestamps.
2. One yellow ball is associated consistently in each view.
3. The side fit predicts a future crossing within the configured horizon.
4. The front fit is evaluated at exactly that crossing timestamp.
5. The reported lateral decision matches labeled validation throws.
6. No arm or gripper API can be reached from the predictor runtime.
