"""Capture cam2 yellow-ball flight on the Viam compute device; read-only.

Requires the existing local machine config for loopback SDK authentication.
Records fresh pixel observations as JSON Lines and predicts an image catch-line
crossing. Pixel acceleration is not physical gravity until 3D calibration exists.
"""

import argparse
import asyncio
import json
import math
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path

from viam.components.camera import Camera
from viam.robot.client import RobotClient

from motion.trajectory_fit import (
    PixelSample, PlaneSample, calibrate_plane_homography, fit_pixel_flight,
    fit_plane_ballistic, pixel_to_plane,
)
from vision.yellow_ball import TemporalPixelTracker, detect_yellow_candidates
from vision.local_camera import credentials, decode_color, positive


async def run(args):
    key_id, key = credentials(args.machine_config)
    options = RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id)
    options.dial_options.disable_webrtc = True
    options.dial_options.timeout = 10
    homography = None
    if args.plane_calibration:
        calibration = json.loads(args.plane_calibration.read_text(encoding="utf-8"))
        homography = calibrate_plane_homography(
            calibration["image_points_px"], calibration["plane_points_mm"])
    record = open(args.record, "a", encoding="utf-8") if args.record else nullcontext(None)
    with record as output:
        async with await RobotClient.at_address("127.0.0.1:8080", options) as robot:
            camera = Camera.from_robot(robot, args.camera)
            history = deque(maxlen=10)
            tracker = TemporalPixelTracker(args.max_gap_s, args.association_gate_px)
            track_id = tracker.track_id
            last_timestamp = None
            shot_id = 0
            started = report_at = time.monotonic()
            last_print = float("-inf")
            next_poll = started
            counts = dict(fresh=0, detected=0, duplicate=0, stale=0, missed=0)
            print(f"Local {args.camera} flight tracker; capture-time pixels only; no arm motion.", flush=True)
            while args.duration is None or time.monotonic() - started < args.duration:
                images, metadata = await camera.get_images(timeout=5)
                timestamp = metadata.captured_at.seconds + metadata.captured_at.nanos / 1e9
                age = time.time() - timestamp
                if not math.isfinite(timestamp) or timestamp <= 0 or not 0 <= age <= args.max_age_s:
                    counts["stale"] += 1
                    history.clear()
                elif last_timestamp is not None and timestamp <= last_timestamp:
                    counts["duplicate"] += 1
                else:
                    last_timestamp = timestamp
                    counts["fresh"] += 1
                    sources = {image.name: image for image in images}
                    if args.source not in sources:
                        raise ValueError(f"Source {args.source!r} missing; available: {list(sources)}")
                    detector_config = dict(
                        hue_min=args.yellow_h_min, hue_max=args.yellow_h_max,
                        saturation_min=args.yellow_s_min, value_min=args.yellow_v_min,
                        min_area_px=args.min_area_px, min_circularity=args.min_circularity)
                    measurement = tracker.update(timestamp, detect_yellow_candidates(
                        decode_color(sources[args.source]), detector_config))
                    if measurement is None:
                        counts["missed"] += 1
                        if not tracker.is_active(timestamp) and history:
                            history.clear()
                    else:
                        counts["detected"] += 1
                        if tracker.track_id != track_id:
                            if track_id:
                                shot_id += 1
                            history.clear()
                            track_id = tracker.track_id
                        center = (measurement.u, measurement.v)
                        sample = PixelSample(timestamp, measurement.u, measurement.v)
                        history.append(sample)
                        plane = pixel_to_plane(*center, homography) if homography is not None else None
                        if output is not None:
                            row = dict(shot_id=shot_id, camera=args.camera,
                                       timestamp=timestamp, u=center[0], v=center[1])
                            if plane is not None:
                                row.update(x_mm=plane[0], z_mm=plane[1])
                            output.write(json.dumps(row) + "\n")
                            output.flush()
                        fit = None
                        if len(history) >= 5:
                            try:
                                fit = fit_pixel_flight(history, max_error_px=args.max_fit_error_px)
                            except ValueError:
                                pass
                        plane_fit = None
                        if homography is not None and len(history) >= 4:
                            plane_samples = [PlaneSample(item.timestamp, *pixel_to_plane(
                                item.u, item.v, homography)) for item in history]
                            try:
                                plane_fit = fit_plane_ballistic(
                                    plane_samples, max_error_mm=args.max_plane_error_mm)
                            except ValueError:
                                pass
                        if time.monotonic() - last_print >= 1 / args.print_hz:
                            line = f"shot={shot_id} t={timestamp:.3f} uv=({center[0]:.1f},{center[1]:.1f}) px"
                            if fit is not None:
                                line += (f" vu={fit.du_px_s:+.0f} px/s "
                                         f"v(t)={fit.v_px:.1f}{fit.dv_px_s:+.1f}*dt"
                                         f"{0.5*fit.d2v_px_s2:+.1f}*dt^2 "
                                         f"rms={fit.rms_error_px:.1f}px")
                                if args.catch_u_px is not None:
                                    try:
                                        crossing_t, crossing_v = fit.crossing(args.catch_u_px, args.max_horizon_s)
                                        line += f" catch_in={crossing_t-timestamp:.3f}s catch_v={crossing_v:.1f}px"
                                    except ValueError:
                                        pass
                            if plane_fit is not None:
                                line += (f" | plane=({plane_fit.x_mm:.0f},{plane_fit.z_mm:.0f})mm "
                                         f"velocity=({plane_fit.vx_mm_s:+.0f},"
                                         f"{plane_fit.vz_mm_s:+.0f})mm/s "
                                         f"gravity={plane_fit.gravity_mm_s2:.0f}mm/s^2 "
                                         f"rms={plane_fit.rms_error_mm:.1f}mm")
                                if args.catch_x_mm is not None:
                                    try:
                                        crossing_t, crossing_z = plane_fit.crossing(
                                            args.catch_x_mm, args.max_horizon_s)
                                        line += (f" physical_catch_in={crossing_t-timestamp:.3f}s "
                                                 f"catch_z={crossing_z:.0f}mm")
                                    except ValueError:
                                        pass
                            print(line, flush=True)
                            last_print = time.monotonic()
                now = time.monotonic()
                if now - report_at >= 2:
                    elapsed = now - report_at
                    print(f"fresh={counts['fresh']/elapsed:.1f}Hz "
                          f"ball={counts['detected']/elapsed:.1f}Hz "
                          f"duplicates={counts['duplicate']} stale={counts['stale']} "
                          f"misses={counts['missed']}", flush=True)
                    counts = dict.fromkeys(counts, 0)
                    report_at = now
                next_poll = max(next_poll + 1 / args.hz, time.monotonic())
                await asyncio.sleep(max(0, next_poll - time.monotonic()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-config", type=Path, required=True,
                        help="Existing Viam cached config on the compute device; never logged")
    parser.add_argument("--camera", default="cam2")
    parser.add_argument("--source", default="color")
    parser.add_argument("--hz", type=positive, default=120.0)
    parser.add_argument("--print-hz", type=positive, default=3.0)
    parser.add_argument("--duration", type=positive)
    parser.add_argument("--max-age-s", type=positive, default=0.25)
    parser.add_argument("--max-gap-s", type=positive, default=0.15)
    parser.add_argument("--max-fit-error-px", type=positive, default=8.0)
    parser.add_argument("--min-area-px", type=positive, default=80.0)
    parser.add_argument("--min-circularity", type=positive, default=0.55)
    parser.add_argument("--association-gate-px", type=positive, default=80.0)
    parser.add_argument("--yellow-h-min", type=int, default=18)
    parser.add_argument("--yellow-h-max", type=int, default=40)
    parser.add_argument("--yellow-s-min", type=int, default=90)
    parser.add_argument("--yellow-v-min", type=int, default=80)
    parser.add_argument("--max-horizon-s", type=positive, default=0.5)
    parser.add_argument("--catch-u-px", type=float, help="Measured catch-line column in cam2 pixels")
    parser.add_argument("--plane-calibration", type=Path,
                        help="JSON with image_points_px and matching vertical-plane points [x,z] mm")
    parser.add_argument("--catch-x-mm", type=float,
                        help="Physical catch-line X in the calibrated throw plane")
    parser.add_argument("--max-plane-error-mm", type=positive, default=30.0)
    parser.add_argument("--record", type=Path, help="Append fresh detections to JSON Lines file")
    args = parser.parse_args()
    if any(value is not None and not math.isfinite(value)
           for value in (args.catch_u_px, args.catch_x_mm)):
        parser.error("Catch-line coordinates must be finite")
    if args.catch_x_mm is not None and args.plane_calibration is None:
        parser.error("--catch-x-mm requires --plane-calibration")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nTracking stopped.")
    except (ValueError, OSError, RuntimeError, asyncio.TimeoutError) as error:
        parser.exit(2, f"Side flight tracker failed: {error}\n")


if __name__ == "__main__":
    main()
