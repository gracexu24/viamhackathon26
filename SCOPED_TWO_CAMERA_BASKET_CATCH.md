# Two-Camera Yellow-Ball Catch Scope

## Objective

- `cam2` supplies the side view and predicts when the ball reaches the fixed
  catch plane.
- The front camera predicts the ball's `(u,v)` pixel position at that timestamp.
- Phase 3 compares that point with the measured basket center and produces one
  validated, two-axis UF850 target within the fixed catch plane.

Yellow-ball detection, camera association, and side-view catch-time prediction
remain separate from robot motion.

## Implemented data flow

```text
cam2 color frames
  -> yellow HSV candidates
  -> temporal association
  -> HELD then RELEASED state transition
  -> clear pre-release history
  -> u(t) linear fit and v(t) quadratic fit
  -> future crossing of catch_u_px
  -> catch_timestamp

front-camera color frames
  -> yellow HSV candidates
  -> temporal association
  -> linear u(t),v(t) fit
  -> evaluate both at catch_timestamp
  -> horizontal_error_px, vertical_error_px

structured prediction
  -> validate age, timing, lead time, and fit quality
  -> independent deadbands, signs, and pixel/mm scales
  -> two configured catch-plane axes
  -> fixed plane-normal coordinate and basket orientation
  -> dry-run or one arm command
```

Both streams use camera capture timestamps. Duplicate, stale, ambiguous, and
poor-fit observations are rejected. Cam2 owns the shared release event; the
front loop clears its history on the same release ID and accepts only real
measurements at or after that release timestamp.

## Current modules

| Path | Responsibility |
| --- | --- |
| `motion/two_camera_rgb_local.py` | Concurrent predictor and structured result. |
| `motion/intercept_controller.py` | Mapping, safety validation, and one arm move. |
| `motion/trajectory_fit.py` | Pixel, throw-plane, and world trajectory fitting. |
| `motion/side_flight_local.py` | Side-camera diagnostics and optional plane fit. |
| `vision/yellow_ball.py` | Yellow HSV detection and temporal association. |
| `vision/local_camera.py` | Read-only local Viam camera helpers. |
| `vision/transforms.py` | Validated rigid transforms for calibrated 3D points. |
| `cam2_transform.json` | Measured `T_world_from_cam2` and Viam frame block. |

## Coordinate frames

The RGB predictor is intentionally pixel-space. A pixel identifies a camera ray,
not an absolute 3D point, so `T_world_from_cam2` cannot be applied until depth or
a calibrated ray/plane intersection provides cam2-frame XYZ.

For calibrated 3D points, use column vectors and millimetres:

```text
p_world = T_world_from_cam2 @ [x_cam2, y_cam2, z_cam2, 1]
```

The homogeneous inverse uses `R.T` and `-R.T @ t`; it is not the transpose of
the entire 4x4 matrix.

## Structured Phase 2 contract

```text
valid
prediction_timestamp
catch_timestamp
time_to_catch_s
predicted_front_u_at_catch
predicted_front_v_at_catch
basket_u_px
basket_v_px
horizontal_error_px
vertical_error_px
side_fit_rms_px
front_fit_rms_px
```

The front U and V predictions are evaluated at exactly the cam2 catch timestamp.
No console parsing is used.

## Phase 3 mapping

```text
horizontal_move_mm = horizontal_error_px * horizontal_mm_per_pixel * horizontal_sign
vertical_move_mm   = vertical_error_px   * vertical_mm_per_pixel   * vertical_sign
```

The two dimensions have independent deadbands, scales, inversion flags, movement
limits, and absolute workspace bounds. The target begins at the verified
catch-ready pose, changes only the two configured in-plane coordinates, forces
the configured plane-normal value, and preserves all orientation fields.

All physical calibration fields in `intercept_controller.config.example.json`
are `null`. They must be measured on the real UF850 setup.

## Safety and execution

- Dry-run is the default; only `--execute` enables arm access.
- A valid live attempt issues at most one `move_to_position` call.
- CENTER in both dimensions holds without a command.
- Invalid, stale, late, non-finite, poor-fit, over-limit, out-of-bounds, or
  already-moving conditions abort before target movement.
- A failed move is stopped once and never retried automatically.
- Continuous servoing, depth fusion, ML, MPC, and streamed joint commands remain
  out of scope.

## Required calibration

- Cam2 catch-line column.
- Front-camera basket-center U and V.
- Catch-ready position and basket orientation.
- Two in-plane UF850 axes and the remaining plane-normal axis/value.
- Direction sign and millimetres per pixel for both image dimensions.
- Safe absolute bounds and maximum correction for both movable axes.
- Conservative measured minimum lead time.
- Yellow HSV/circularity thresholds for actual lighting.

## Testing order

1. Run mapping, deadband, inversion, scale, clamp, bounds, and timing unit tests.
2. Run synthetic predictions with no arm access.
3. Verify center/left/right/up/down/diagonal poses without a ball.
4. Calibrate both pixel-to-millimetre scales and direction signs.
5. Run live throws in dry-run mode and compare proposed targets visually.
6. Only then run a controlled live attempt with `--execute`.

## Acceptance criteria

1. Both cameras deliver fresh, increasing capture timestamps.
2. Cam2 predicts a future crossing within the configured horizon.
3. The front fit predicts U and V at that exact timestamp.
4. Both image errors map to the intended robot-plane directions.
5. The final plane-normal coordinate and orientation remain unchanged.
6. Dry-run never reaches an arm resource.
7. Valid execution sends exactly one movement command.
8. Failed movement does not retry.
9. Physical calibration verifies the safe region and measured lead time.
