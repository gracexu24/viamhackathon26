# Viam Two-Camera Ball Catch

This repository contains a two-camera yellow-ball predictor and a one-shot,
two-axis UF850 basket controller. `cam2` predicts when the ball reaches a fixed
catch plane. The front camera predicts the ball's `(u, v)` pixel position at
that exact timestamp. Phase 3 converts the two image errors into one configured
move within the catch plane.

The default is always read-only. Physical movement requires `--execute`.

## Current pipeline

```text
cam2 yellow-ball track -> catch-plane timestamp
                                  |
front-camera track -> predicted (u,v) at that timestamp
                                  |
                     basket-center pixel error
                                  |
                  validated two-axis arm target
```

`motion/two_camera_rgb_local.py` contains perception and imports no arm or
motion API. `motion/intercept_controller.py` owns the small, one-shot hardware
call and consumes the structured Phase 2 result directly.

## Setup

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For a direct cloud connection, create `.env` from Viam credentials. Do not
commit it:

```dotenv
VIAM_ADDRESS=your-machine-address
VIAM_API_KEY_ID=your-api-key-id
VIAM_API_KEY=your-api-key
```

Code running beside `viam-server` instead uses the cached machine configuration
and connects locally at `127.0.0.1:8080`.

## Run the read-only predictor

Measure the cam2 catch column and the front-camera basket center first:

```bash
cd /opt/viam/trajectory-local
/opt/viam/trajectory-local-venv/bin/python -m motion.two_camera_rgb_local \
  --machine-config /path/to/cached-machine-config.json \
  --side-camera cam2 \
  --front-camera cam \
  --catch-u-px "$CATCH_U_PX" \
  --basket-u-px "$BASKET_U_PX" \
  --basket-v-px "$BASKET_V_PX" \
  --duration 20
```

Use `--front-camera cam1` if that is the installed resource name. The result
contains catch time, predicted front `(u,v)`, basket `(u,v)`, both signed image
errors, and side/front fit quality.

The side camera first confirms that the ball is held, then emits one release
event after sustained fast motion. Both camera histories are cleared at that
event, so trajectory fits contain only real post-release measurements. Tune
this behavior with `--held-speed-threshold-px-s`,
`--release-speed-threshold-px-s`, `--release-min-consecutive-frames`, and
`--held-min-duration-s`. Predicted tracker positions are used only for
association and never enter release detection or trajectory fitting.

## Cam2 transform

The measured fixed-camera calibration is stored in `cam2_transform.json` as
`T_world_from_cam2`, using column vectors and millimetres:

```text
p_world = T_world_from_cam2 @ [x_cam2, y_cam2, z_cam2, 1]
```

Transform a known cam2-frame 3D point with:

```bash
python -m vision.transforms --config cam2_transform.json --point X Y Z
```

The full inverse is not the transpose of the 4x4 matrix. `RigidTransform.inverse()`
uses `R.T` for rotation and `-R.T @ t` for translation. The RGB predictor remains
pixel-space and does not apply this transform to pixels, because a pixel is a ray
rather than a camera-frame 3D point.

## Phase 3 one-shot catch-plane movement

Copy `intercept_controller.config.example.json` to
`intercept_controller.config.json` and replace every `null` with a physically
measured value. The example is deliberately non-runnable: axes, signs, scales,
pose, bounds, plane coordinate, and lead time must not be guessed.

Integrated live dry run:

```bash
/opt/viam/trajectory-local-venv/bin/python -m motion.intercept_controller \
  --config intercept_controller.config.json \
  --machine-config /path/to/cached-machine-config.json \
  --arm arm \
  --side-camera cam2 \
  --front-camera cam \
  --catch-u-px "$CATCH_U_PX" \
  --basket-u-px "$BASKET_U_PX" \
  --basket-v-px "$BASKET_V_PX"
```

Without `--execute`, the controller prints `WOULD_MOVE` and never obtains an arm
resource. After dry-run and supervised no-ball calibration, add `--execute` to
allow exactly one `move_to_position` command. Restart the command for each catch
attempt.

The controller validates prediction validity, age, timing, lead time, fit errors,
independent deadbands, movement limits, rectangular in-plane bounds, fixed plane
normal, fixed orientation, and idle arm state. An execution failure is stopped
and is never retried automatically.

## Safety and calibration

- Verify the catch-ready pose, two in-plane axes, plane-normal axis/value, both
  direction signs, and both millimetre-per-pixel scales on the real setup.
- Test center, left, right, up, down, diagonal, and center again without a ball.
- Verify collision geometry, reachable bounds, exclusive arm control, and an
  accessible emergency stop before using `--execute`.
- The cam2 transform helps describe the fixed camera in `world`; it does not by
  itself determine the front-camera pixel-to-arm mapping.

## Tests

```bash
python -m unittest discover -s tests -v
```

The unit tests are headless and use mocked arm hardware.

## Project layout

| Path | Purpose |
| --- | --- |
| `motion/two_camera_rgb_local.py` | Concurrent two-camera predictor. |
| `motion/intercept_controller.py` | Validated one-shot catch-plane controller. |
| `motion/trajectory_fit.py` | Pixel-, plane-, and world-space trajectory fits. |
| `motion/side_flight_local.py` | Side-camera diagnostics and optional plane fit. |
| `vision/yellow_ball.py` | Yellow-ball detection and temporal association. |
| `vision/local_camera.py` | Read-only local Viam camera helpers. |
| `vision/transforms.py` | Validated rigid camera/world transforms. |
| `cam2_transform.json` | Measured `T_world_from_cam2` calibration. |
| `tests/` | Unit tests with synthetic perception and mocked hardware. |
