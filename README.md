# Viam Two-Camera Ball Catch

This repository contains perception and trajectory-estimation building blocks
for catching a ball with a Viam-controlled arm. The current two-camera pipeline
tracks a yellow ball in RGB streams, predicts where it will cross a configured
catch line, and classifies that crossing as `LEFT`, `CENTER`, or `RIGHT`.

The predictor is read-only. It does not command the arm, and there is not yet an
integrated real-time catch controller.

## Current pipeline

```text
side camera (cam2) -> yellow-ball samples -> parabolic side fit
                                              |
                                              v
                                    catch-line timestamp
                                              |
front camera (cam) -> yellow-ball samples -> lateral fit -> LEFT/CENTER/RIGHT
```

`motion/two_camera_rgb_local.py` opens the two cameras, tracks the ball, and
prints predictions without importing or calling robot-motion APIs.

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For direct cloud connections, create `.env` from your Viam machine credentials:

```dotenv
VIAM_ADDRESS=your-machine-address
VIAM_API_KEY_ID=your-api-key-id
VIAM_API_KEY=your-api-key
```

Do not commit `.env`. Code running on the Viam machine can instead use its cached
machine configuration and connect to the local server at `127.0.0.1:8080`.

## Run the two-camera predictor

First measure these pixel coordinates from the installed camera views:

- `--catch-u-px`: the side-camera horizontal pixel where the ball reaches the
  catch plane.
- `--basket-u-px`: the front-camera horizontal pixel at the basket center.

On the Viam compute module, a typical invocation is:

```bash
cd /opt/viam/trajectory-local
/opt/viam/trajectory-local-venv/bin/python -m motion.two_camera_rgb_local \
  --machine-config /root/.viam/cached_cloud_config_49d63d4f-4191-4d94-9e43-e379f846dd0e.json \
  --side-camera cam2 \
  --front-camera cam \
  --catch-u-px 700 \
  --basket-u-px 424 \
  --duration 20
```

The side view fits the observed flight in pixel space and estimates when the
ball will cross the catch line. The front view predicts its horizontal position
at that same capture timestamp. The final output reports `LEFT`, `CENTER`, or
`RIGHT` relative to the configured basket center. Camera names and thresholds
can be adjusted with `python -m motion.two_camera_rgb_local --help`.

## Safety and limitations

- The current two-camera command is prediction-only; it does not move hardware.
- Its output is pixel-relative and does not provide a world-frame 3D arm target.
- Do not treat this prototype as a time-synchronized catch controller.
- Run legacy motion experiments only with exclusive arm control, verified Viam
  collision geometry, conservative workspace bounds, and an emergency stop.

## Supporting tools

| Path | Purpose |
| --- | --- |
| `motion/trajectory_fit.py` | Pixel-, plane-, and world-space trajectory models and fitting helpers. |
| `motion/side_flight_local.py` | Read-only side-camera flight tracking and plane calibration. |
| `motion/trajectory_local.py` | Shared local tracking helpers and diagnostic trajectory runner. |
| `vision/yellow_ball.py` | Yellow-ball detection and temporal pixel tracking. |
| `vision/side_tracker.py` | Side-camera tracking diagnostics. |
| `vision/relative_3d.py` | Relative 3D camera diagnostics. |
| `motion/viam_stationary_grab.py` | Earlier stationary-object grab workflow retained for reference. |
| `motion/viam_ball_catch.py` | Earlier Viam-native catch experiment retained for reference. |

Detailed design notes are in `SCOPED_TWO_CAMERA_BASKET_CATCH.md`. Older scoped
notes and configuration examples remain where they are still useful to tested
diagnostic or reference workflows.

## Tests

Run the complete unit-test suite with:

```bash
python -m unittest discover -s tests -v
```

The live desktop preview and its two import-level tests require a Python
installation with Tk support. Headless perception and trajectory-fitting tests
do not require a display.

## Project layout

```text
motion/       trajectory fitting, prediction, and legacy Viam motion workflows
vision/       camera detection and diagnostic utilities
tests/        unit tests with mocked Viam hardware
*.json        calibration examples
connection.py shared Viam connection helpers
```
