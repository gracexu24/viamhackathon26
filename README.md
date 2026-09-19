# Viam Two-Camera Ball Catch

This repository contains the perception and motion building blocks for catching a
ball with a Viam-controlled arm. The current pipeline tracks a yellow ball in two
RGB camera streams, predicts where it will cross a configured catch line, and
classifies that crossing as `LEFT`, `CENTER`, or `RIGHT`. A separate motion tool
can move the arm to the closest point on a supplied 3D trajectory.

The two pieces are intentionally separate today: the camera predictor does not
yet produce a world-frame 3D trajectory, and there is no integrated real-time
catch controller.

## Current pipeline

```text
side camera (cam2) -> yellow-ball samples -> parabolic side fit
                                              |
                                              v
                                    catch-line timestamp
                                              |
front camera (cam) -> yellow-ball samples -> lateral fit -> LEFT/CENTER/RIGHT

3D trajectory JSON or WorldFlight -> closest world-frame point -> Viam Motion move
```

`motion/two_camera_rgb_local.py` is read-only: it opens the two cameras, tracks
the ball, and prints a prediction. It does not command the arm. Arm motion is
handled separately by `motion/move_to_trajectory.py` and defaults to preview
mode.

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

First measure these pixel coordinates from your installed camera views:

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

## Move the arm toward a 3D trajectory

`motion/move_to_trajectory.py` reads the arm flange pose from Viam, finds the
geometrically closest point on a world-frame trajectory, preserves the current
flange orientation, and asks Viam Motion for one planned move.

It accepts either a sampled polyline:

```json
{
  "reference_frame": "world",
  "points": [[100, 0, 300], [250, 40, 350], [400, 80, 300]]
}
```

or a quadratic flight model:

```json
{
  "reference_frame": "world",
  "timestamp": 0.0,
  "position_mm": [100, 0, 300],
  "velocity_mm_s": [500, 100, 600],
  "acceleration_mm_s2": [0, 0, -9810],
  "rms_error_mm": 8.0
}
```

Preview a target without moving hardware:

```bash
python -m motion.move_to_trajectory \
  --trajectory trajectory.example.json
```

Execute only after supplying an explicit allowed workspace:

```bash
python -m motion.move_to_trajectory \
  --trajectory trajectory.json \
  --workspace '[[0,-1000,0],[600,100,700]]' \
  --log-file move-to-trajectory.log \
  --execute
```

The workspace is an axis-aligned box described by minimum and maximum XYZ
corners in millimetres. Targets outside it are rejected. The command logs the
input, current pose, chosen target, requested move, result, and failures.

This tool selects the nearest geometric point only; it does not synchronize the
arm's arrival with the ball. The Python Motion API also does not expose a
per-call “maximum speed” switch, so actual speed and acceleration remain subject
to the arm driver, Viam configuration, planning, and collision constraints.

## Safety

- Run in preview mode before every new trajectory or workspace.
- Keep people and loose objects outside the robot workspace.
- Configure Viam frame-system geometry and collision constraints for the real
  installation.
- Give only one process control of the arm at a time.
- Use an emergency stop and conservative workspace bounds during testing.
- Do not treat this prototype as a time-synchronized catch controller.

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
installation with Tk support. Headless perception, fitting, and motion-planning
tests do not require a display.

## Project layout

```text
motion/       trajectory fitting, prediction, and Viam motion commands
vision/       camera detection and diagnostic utilities
tests/        unit tests with mocked Viam hardware
*.json        calibration and trajectory examples
connection.py shared Viam connection helpers
```
