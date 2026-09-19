"""Read-only two-camera yellow-ball catch predictor; never moves hardware.

cam2 predicts when the ball crosses a configured side-image column. The front
camera predicts horizontal ball position at that same capture timestamp and
reports LEFT/CENTER/RIGHT relative to a configured basket-center pixel.
"""

import argparse
import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from viam.components.camera import Camera
from viam.robot.client import RobotClient

from motion.trajectory_fit import (
    FrontSample, PixelSample, fit_front_lateral, fit_pixel_flight, lateral_decision,
)
from vision.local_camera import credentials, decode_color, positive
from vision.yellow_ball import TemporalPixelTracker, detect_yellow_candidates


@dataclass(frozen=True)
class SideCatchPrediction:
    catch_timestamp: float
    side_timestamp: float
    side_samples: int
    side_u: float
    side_v: float
    side_velocity_u_px_s: float
    side_rms_px: float
    side_catch_v_px: float


@dataclass(frozen=True)
class CatchPrediction:
    catch_timestamp: float
    time_to_catch_s: float
    side_samples: int
    side_fit_rms_px: float
    side_velocity_u_px_s: float
    front_samples: int
    front_fit_rms_px: float
    front_velocity_u_px_s: float
    predicted_ball_u_at_catch: float
    basket_u_px: float
    lateral_error_px: float
    decision: str


@dataclass
class SharedState:
    side_catch: SideCatchPrediction | None = None
    stopped: bool = False


def captured_at_seconds(metadata):
    value = metadata.captured_at
    return value.seconds + value.nanos / 1e9


def predict_side_catch(samples, catch_u_px, max_horizon_s, max_fit_error_px):
    """Fit real side samples and return their next crossing of catch_u_px."""
    fit = fit_pixel_flight(samples, max_error_px=max_fit_error_px)
    catch_timestamp, catch_v = fit.crossing(catch_u_px, max_horizon_s)
    if catch_timestamp <= samples[-1].timestamp:
        raise ValueError("Side crossing is not in the future")
    return SideCatchPrediction(
        catch_timestamp=catch_timestamp,
        side_timestamp=float(samples[-1].timestamp),
        side_samples=len(samples),
        side_u=float(samples[-1].u),
        side_v=float(samples[-1].v),
        side_velocity_u_px_s=fit.du_px_s,
        side_rms_px=fit.rms_error_px,
        side_catch_v_px=catch_v,
    )


def combine_prediction(side_catch, front_samples, front_fit, basket_u_px,
                       center_deadband_px, invert_lateral=False):
    """Evaluate the front fit at the side-camera crossing timestamp."""
    time_to_catch = side_catch.catch_timestamp - front_fit.timestamp
    if time_to_catch < 0:
        raise ValueError("Catch timestamp is behind the front-camera track")
    predicted_u = front_fit.predict_u(side_catch.catch_timestamp)
    error = predicted_u - float(basket_u_px)
    return CatchPrediction(
        catch_timestamp=side_catch.catch_timestamp,
        time_to_catch_s=time_to_catch,
        side_samples=side_catch.side_samples,
        side_fit_rms_px=side_catch.side_rms_px,
        side_velocity_u_px_s=side_catch.side_velocity_u_px_s,
        front_samples=int(front_samples),
        front_fit_rms_px=front_fit.rms_error_px,
        front_velocity_u_px_s=front_fit.du_px_s,
        predicted_ball_u_at_catch=predicted_u,
        basket_u_px=float(basket_u_px),
        lateral_error_px=error,
        decision=lateral_decision(error, center_deadband_px, invert_lateral),
    )


async def side_loop(camera, args, detector_config, shared):
    tracker = TemporalPixelTracker(args.max_gap_s, args.association_gate_px)
    history = deque(maxlen=max(12, args.min_side_samples * 2))
    track_id = tracker.track_id
    last_timestamp = None
    reported_ready = False
    next_poll = time.monotonic()
    while not shared.stopped:
        images, metadata = await camera.get_images(timeout=5)
        timestamp = captured_at_seconds(metadata)
        age = time.time() - timestamp
        if last_timestamp is not None and timestamp <= last_timestamp:
            await asyncio.sleep(0)
            continue
        last_timestamp = timestamp
        sources = {image.name: image for image in images}
        if args.color_source not in sources:
            raise ValueError(f"Side color source missing; available: {list(sources)}")
        color = decode_color(sources[args.color_source])
        if not reported_ready:
            print(f"side camera ready: shape={color.shape[1]}x{color.shape[0]} "
                  f"capture_age={age:.3f}s", flush=True)
            reported_ready = True
        candidates = [] if not 0 <= age <= args.max_age_s else detect_yellow_candidates(
            color, detector_config)
        measurement = tracker.update(timestamp, candidates)
        if measurement is not None:
            if tracker.track_id != track_id:
                history.clear()
                track_id = tracker.track_id
            history.append(PixelSample(measurement.timestamp, measurement.u, measurement.v))
            shared.side_catch = None
            if len(history) >= args.min_side_samples:
                try:
                    shared.side_catch = predict_side_catch(
                        history, args.catch_u_px, args.max_horizon_s,
                        args.max_side_fit_error_px)
                except ValueError:
                    shared.side_catch = None
        elif not tracker.is_active(timestamp):
            history.clear()
            shared.side_catch = None
        next_poll = max(next_poll + 1/args.poll_hz, time.monotonic())
        await asyncio.sleep(max(0, next_poll-time.monotonic()))


async def front_loop(camera, args, detector_config, shared):
    tracker = TemporalPixelTracker(args.max_gap_s, args.association_gate_px)
    history = deque(maxlen=max(12, args.min_front_samples * 2))
    track_id = tracker.track_id
    last_timestamp = None
    reported_ready = False
    last_print = float("-inf")
    next_poll = time.monotonic()
    while not shared.stopped:
        images, metadata = await camera.get_images(timeout=5)
        timestamp = captured_at_seconds(metadata)
        age = time.time() - timestamp
        if last_timestamp is not None and timestamp <= last_timestamp:
            await asyncio.sleep(0)
            continue
        last_timestamp = timestamp
        sources = {image.name: image for image in images}
        if args.color_source not in sources:
            raise ValueError(f"Front color source missing; available: {list(sources)}")
        color = decode_color(sources[args.color_source])
        if not reported_ready:
            print(f"front camera ready: shape={color.shape[1]}x{color.shape[0]} "
                  f"capture_age={age:.3f}s", flush=True)
            reported_ready = True
        candidates = [] if not 0 <= age <= args.max_age_s else detect_yellow_candidates(
            color, detector_config)
        measurement = tracker.update(timestamp, candidates)
        if measurement is not None:
            if tracker.track_id != track_id:
                history.clear()
                track_id = tracker.track_id
            history.append(FrontSample(measurement.timestamp, measurement.u))
        elif not tracker.is_active(timestamp):
            history.clear()
        side_catch = shared.side_catch
        if side_catch is not None and side_catch.catch_timestamp <= timestamp:
            shared.side_catch = None
            side_catch = None
        if side_catch is not None and len(history) >= args.min_front_samples:
            try:
                fit = fit_front_lateral(history, min_samples=args.min_front_samples,
                                        max_error_px=args.max_front_fit_error_px)
                prediction_horizon = side_catch.catch_timestamp - fit.timestamp
                if not 0 <= prediction_horizon <= args.max_horizon_s:
                    raise ValueError("Front prediction horizon is invalid")
                prediction = combine_prediction(
                    side_catch, len(history), fit, args.basket_u_px,
                    args.center_deadband_px, args.invert_lateral)
                if time.monotonic()-last_print >= 1/args.print_hz:
                    print(
                        f"side_samples={prediction.side_samples} "
                        f"side_vu={prediction.side_velocity_u_px_s:+.1f}px/s "
                        f"side_rms={prediction.side_fit_rms_px:.1f}px "
                        f"catch_timestamp={prediction.catch_timestamp:.3f} "
                        f"catch_in={prediction.time_to_catch_s:.3f}s | "
                        f"front_samples={prediction.front_samples} "
                        f"front_vu={prediction.front_velocity_u_px_s:+.1f}px/s "
                        f"front_rms={prediction.front_fit_rms_px:.1f}px "
                        f"predicted_u={prediction.predicted_ball_u_at_catch:.1f}px "
                        f"basket_u={prediction.basket_u_px:.1f}px "
                        f"error={prediction.lateral_error_px:+.1f}px "
                        f"decision={prediction.decision}", flush=True)
                    last_print = time.monotonic()
            except ValueError:
                pass
        next_poll = max(next_poll + 1/args.poll_hz, time.monotonic())
        await asyncio.sleep(max(0, next_poll-time.monotonic()))


async def run(args):
    key_id, key = credentials(args.machine_config)
    options = RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id)
    options.dial_options.disable_webrtc = True
    options.dial_options.timeout = 10
    detector_config = {
        "hue_min": args.yellow_h_min,
        "hue_max": args.yellow_h_max,
        "saturation_min": args.yellow_s_min,
        "value_min": args.yellow_v_min,
        "min_area_px": args.min_area_px,
        "min_circularity": args.min_circularity,
    }
    shared = SharedState()
    async with await RobotClient.at_address("127.0.0.1:8080", options) as robot:
        side = Camera.from_robot(robot, args.side_camera)
        front = Camera.from_robot(robot, args.front_camera)
        print(f"Read-only RGB predictor: side={args.side_camera!r} "
              f"front={args.front_camera!r}; no robot motion.", flush=True)
        tasks = [asyncio.create_task(side_loop(side, args, detector_config, shared)),
                 asyncio.create_task(front_loop(front, args, detector_config, shared))]
        try:
            if args.duration is None:
                await asyncio.gather(*tasks)
            else:
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=args.duration)
        except asyncio.TimeoutError:
            pass
        finally:
            shared.stopped = True
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-config", type=Path, required=True)
    parser.add_argument("--side-camera", default="cam2")
    parser.add_argument("--front-camera", default="cam")
    parser.add_argument("--color-source", default="color")
    parser.add_argument("--catch-u-px", type=float, required=True)
    parser.add_argument("--basket-u-px", type=float, required=True)
    parser.add_argument("--center-deadband-px", type=positive, default=25.0)
    parser.add_argument("--min-side-samples", type=int, default=7)
    parser.add_argument("--min-front-samples", type=int, default=5)
    parser.add_argument("--max-gap-s", type=positive, default=0.1)
    parser.add_argument("--max-horizon-s", type=positive, default=0.6)
    parser.add_argument("--invert-lateral", action="store_true")
    parser.add_argument("--duration", type=positive)
    parser.add_argument("--print-hz", type=positive, default=5.0)
    parser.add_argument("--poll-hz", type=positive, default=120.0)
    parser.add_argument("--max-age-s", type=positive, default=0.25)
    parser.add_argument("--association-gate-px", type=positive, default=80.0)
    parser.add_argument("--max-side-fit-error-px", "--max-side-error-px",
                        dest="max_side_fit_error_px", type=positive, default=8.0)
    parser.add_argument("--max-front-fit-error-px", "--max-front-error-px",
                        dest="max_front_fit_error_px", type=positive, default=8.0)
    parser.add_argument("--yellow-h-min", type=int, default=18)
    parser.add_argument("--yellow-h-max", type=int, default=40)
    parser.add_argument("--yellow-s-min", type=int, default=90)
    parser.add_argument("--yellow-v-min", type=int, default=80)
    parser.add_argument("--min-area-px", type=positive, default=60.0)
    parser.add_argument("--min-circularity", type=positive, default=0.55)
    args = parser.parse_args()
    if args.min_side_samples < 5:
        parser.error("--min-side-samples must be at least 5")
    if args.min_front_samples < 2:
        parser.error("--min-front-samples must be at least 2")
    if not all(math.isfinite(value) for value in (args.catch_u_px, args.basket_u_px)):
        parser.error("Catch and basket pixels must be finite")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nPredictor stopped.")
    except (ValueError, OSError, RuntimeError, asyncio.TimeoutError) as error:
        parser.exit(2, f"Two-camera predictor failed: {error}\n")


if __name__ == "__main__":
    main()
