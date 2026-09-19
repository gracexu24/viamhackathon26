"""Read-only red-ball tracking from the fixed side camera (cam2 by default).

Run ``python -m vision.side_tracker --duration 10`` for a bounded text check,
or add ``--display`` for an annotated live view. This module never moves the arm.
"""

import argparse
import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from viam.components.camera import Camera

from connection import connect


@dataclass(frozen=True)
class SideObservation:
    timestamp: float
    u: float
    v: float
    radius_px: float


def detect_red_ball(color, timestamp):
    """Locate one circular red blob in a BGR frame; return pixel coordinates."""
    if color is None or color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("Expected a BGR color frame")
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 70), (10, 255, 255))
    mask |= cv2.inRange(hsv, (170, 100, 70), (179, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < 80 or perimeter <= 0 or 4 * math.pi * area / perimeter**2 < 0.65:
            continue
        (u, v), radius = cv2.minEnclosingCircle(contour)
        if u - radius <= 0 or v - radius <= 0 or u + radius >= color.shape[1] - 1 \
                or v + radius >= color.shape[0] - 1:
            continue
        candidates.append(SideObservation(timestamp, float(u), float(v), float(radius)))
    return candidates[0] if len(candidates) == 1 else None


def decode_color(image):
    color = cv2.imdecode(np.frombuffer(image.data, np.uint8), cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError("Camera color image is not decodable JPEG/PNG")
    return color


def capture_time(metadata):
    captured = metadata.captured_at
    return captured.seconds + captured.nanos / 1e9


async def track(machine, args):
    camera = Camera.from_robot(machine, args.camera)
    previous = None
    last_timestamp = None
    trail = deque(maxlen=24)
    counts = dict(fresh=0, detected=0, duplicate=0, stale=0, missed=0)
    latest_age_ms = None
    started = report_at = time.monotonic()
    next_poll = started
    last_print = float("-inf")
    print(f"Tracking red ball from {args.camera!r} ({args.source!r}); no robot motion.", flush=True)
    try:
        while args.duration is None or time.monotonic() - started < args.duration:
            images, metadata = await camera.get_images(timeout=5)
            sources = {image.name: image for image in images}
            if args.source not in sources:
                raise ValueError(f"Source {args.source!r} missing; available: {list(sources)}")
            color = decode_color(sources[args.source])
            timestamp = capture_time(metadata)
            age = time.time() - timestamp
            latest_age_ms = age * 1000 if math.isfinite(age) else None
            observation = None
            status = "stale"
            if not math.isfinite(timestamp) or timestamp <= 0 or not 0 <= age <= args.max_age_s:
                counts["stale"] += 1
                previous = None
                trail.clear()
            elif last_timestamp is not None and timestamp <= last_timestamp:
                counts["duplicate"] += 1
                status = "duplicate"
            else:
                status = "fresh"
                last_timestamp = timestamp
                counts["fresh"] += 1
                observation = detect_red_ball(color, timestamp)
                velocity = None
                if observation is None:
                    counts["missed"] += 1
                    previous = None
                    trail.clear()
                else:
                    counts["detected"] += 1
                    trail.append((round(observation.u), round(observation.v)))
                    if previous is not None:
                        dt = timestamp - previous.timestamp
                        if 0 < dt <= 0.25:
                            velocity = ((observation.u - previous.u) / dt,
                                        (observation.v - previous.v) / dt)
                    previous = observation
                    if time.monotonic() - last_print >= 1 / args.print_hz:
                        speed = "" if velocity is None else f" velocity=({velocity[0]:+.1f}, {velocity[1]:+.1f}) px/s"
                        print(f"t={timestamp:.3f} age={age*1000:.0f} ms "
                              f"ball=({observation.u:.1f}, {observation.v:.1f}) px "
                              f"radius={observation.radius_px:.1f} px{speed}", flush=True)
                        last_print = time.monotonic()
            if args.display:
                for a, b in zip(trail, list(trail)[1:]):
                    cv2.line(color, a, b, (255, 255, 0), 2)
                if observation is not None:
                    cv2.circle(color, trail[-1], round(observation.radius_px), (0, 255, 0), 2)
                    cv2.drawMarker(color, trail[-1], (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
                age_text = "unknown" if latest_age_ms is None else f"{latest_age_ms:.0f}ms"
                cv2.putText(color, f"cam={args.camera} {status} age={age_text}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Side camera ball tracker", color)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            now = time.monotonic()
            if now - report_at >= 2:
                elapsed = now - report_at
                age_text = "unknown" if latest_age_ms is None else f"{latest_age_ms:.0f}"
                print(f"fresh={counts['fresh']/elapsed:.1f} Hz "
                      f"ball={counts['detected']/elapsed:.1f} Hz "
                      f"duplicates={counts['duplicate']} stale={counts['stale']} "
                      f"misses={counts['missed']} latest_age_ms={age_text}", flush=True)
                counts = dict.fromkeys(counts, 0)
                report_at = now
            next_poll = max(next_poll + 1 / args.hz, time.monotonic())
            await asyncio.sleep(max(0, next_poll - time.monotonic()))
    finally:
        if args.display:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="cam2", help="Viam side camera resource")
    parser.add_argument("--source", default="color", help="Color source from camera.get_images")
    parser.add_argument("--duration", type=float, help="Stop after this many seconds; otherwise Ctrl+C")
    parser.add_argument("--hz", type=float, default=60, help="Maximum image poll rate")
    parser.add_argument("--print-hz", type=float, default=5, help="Maximum detection print rate")
    parser.add_argument("--max-age-s", type=float, default=0.5, help="Reject older capture timestamps")
    parser.add_argument("--display", action="store_true", help="Show annotated color frames and ball trail")
    args = parser.parse_args()
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.hz, args.print_hz, args.max_age_s)) or (
            args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0)):
        parser.error("Rates, max age, and duration must be positive finite numbers")
    try:
        async def run():
            async with await connect() as machine:
                await track(machine, args)
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nTracking stopped.")
    except (ValueError, OSError, RuntimeError, asyncio.TimeoutError) as error:
        parser.exit(2, f"Side tracker failed: {error}\n")


if __name__ == "__main__":
    main()
