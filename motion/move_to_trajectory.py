"""Move the arm flange to the closest point on a sampled 3D trajectory.

The trajectory source is intentionally simple until live ball prediction exists:
pass a JSON file containing either a list of ``[x, y, z]`` points or an object
with ``{"reference_frame": "world", "points": [...]}``. Coordinates are mm.

Preview is the default. Add ``--execute`` to command one unconstrained Viam
Motion service move. The arm's configured motion profile controls physical
velocity; this program adds no sleeps, waypoints, or velocity restriction.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Iterable, Sequence

import numpy as np
from viam.components.arm import Arm
from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient

from connection import connect
from motion.trajectory_fit import WorldFlight


LOGGER = logging.getLogger("move_to_trajectory")


@dataclass(frozen=True)
class ClosestPoint:
    """A point on either a polyline or fitted ballistic trajectory."""

    xyz: tuple[float, float, float]
    distance_mm: float
    segment_index: int | None = None
    segment_fraction: float | None = None
    trajectory_time_s: float | None = None
    trajectory_timestamp: float | None = None


def _xyz(values: Iterable[float], name: str) -> tuple[float, float, float]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain three numbers") from error
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain three finite numbers")
    return result


def validate_trajectory(
    trajectory: Iterable[Iterable[float]],
) -> tuple[tuple[float, float, float], ...]:
    """Return a finite trajectory containing at least one line segment."""
    try:
        points = tuple(_xyz(point, f"trajectory point {index}")
                       for index, point in enumerate(trajectory))
    except TypeError as error:
        raise ValueError("trajectory must be an iterable of XYZ points") from error
    if len(points) < 2:
        raise ValueError("trajectory must contain at least two XYZ points")
    return points


def closest_point_on_trajectory(
    end_effector_xyz: Sequence[float],
    trajectory: Iterable[Iterable[float]],
) -> ClosestPoint:
    """Find the Euclidean closest point on a 3D piecewise-linear trajectory.

    Projection is clamped to each finite segment. Duplicate consecutive points
    are treated as zero-length segments rather than causing a division by zero.
    This is purely geometric: timestamps and arm travel time are not considered.
    """
    query = _xyz(end_effector_xyz, "end-effector XYZ")
    points = validate_trajectory(trajectory)
    best: ClosestPoint | None = None

    for index, (start, end) in enumerate(zip(points, points[1:])):
        delta = tuple(b - a for a, b in zip(start, end))
        length_squared = sum(value * value for value in delta)
        if length_squared <= 1e-12:
            fraction = 0.0
        else:
            fraction = sum((q - a) * d for q, a, d in zip(query, start, delta)) / length_squared
            fraction = min(1.0, max(0.0, fraction))
        projected = tuple(a + fraction * d for a, d in zip(start, delta))
        distance = math.dist(query, projected)
        candidate = ClosestPoint(projected, distance, index, fraction)
        if best is None or candidate.distance_mm < best.distance_mm:
            best = candidate

    assert best is not None  # validate_trajectory guarantees at least one segment.
    return best


def closest_point_on_world_flight(
    end_effector_xyz: Sequence[float],
    trajectory: WorldFlight,
    *,
    horizon_s: float = 0.5,
) -> ClosestPoint:
    """Find the exact closest point on a gravity-constrained ``WorldFlight``.

    Squared distance from a point to a quadratic flight path has a cubic
    derivative. Its real roots inside the prediction horizon, plus both horizon
    endpoints, contain the global minimum over that finite interval.
    """
    query = np.asarray(_xyz(end_effector_xyz, "end-effector XYZ"), dtype=float)
    position = np.asarray(_xyz(trajectory.position_mm, "trajectory position"), dtype=float)
    velocity = np.asarray(_xyz(trajectory.velocity_mm_s, "trajectory velocity"), dtype=float)
    acceleration = np.asarray(
        _xyz(trajectory.acceleration_mm_s2, "trajectory acceleration"), dtype=float
    )
    if not math.isfinite(trajectory.timestamp):
        raise ValueError("trajectory timestamp must be finite")
    if not math.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError("trajectory horizon must be positive and finite")

    relative = position - query
    # For r(t)=relative+velocity*t+0.5*acceleration*t^2, solve
    # r(t) dot r'(t)=0. Leading near-zero coefficients are removed so constant-
    # velocity and stationary paths work through the same implementation.
    coefficients = np.asarray([
        0.5 * acceleration @ acceleration,
        1.5 * velocity @ acceleration,
        velocity @ velocity + relative @ acceleration,
        relative @ velocity,
    ], dtype=float)
    nonzero = np.flatnonzero(np.abs(coefficients) > 1e-12)
    candidates = [0.0, float(horizon_s)]
    if nonzero.size:
        for root in np.roots(coefficients[nonzero[0]:]):
            if abs(float(root.imag)) <= 1e-8 and 0 <= float(root.real) <= horizon_s:
                candidates.append(float(root.real))

    best = None
    for seconds in candidates:
        timestamp = float(trajectory.timestamp) + seconds
        xyz = _xyz(trajectory.predict(timestamp), "predicted trajectory point")
        candidate = ClosestPoint(
            xyz=xyz,
            distance_mm=math.dist(query, xyz),
            trajectory_time_s=seconds,
            trajectory_timestamp=timestamp,
        )
        if best is None or candidate.distance_mm < best.distance_mm:
            best = candidate
    assert best is not None
    return best


def load_trajectory(
    path: str | Path, expected_frame: str = "world"
) -> tuple[tuple[float, float, float], ...] | WorldFlight:
    """Load a sampled polyline or ``trajectory_fit.WorldFlight`` from JSON."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read trajectory JSON {source}: {error}") from error

    if isinstance(payload, dict):
        frame = payload.get("reference_frame", expected_frame)
        if frame != expected_frame:
            raise ValueError(
                f"Trajectory frame {frame!r} does not match requested world frame {expected_frame!r}"
            )
        if "points" in payload:
            payload = payload["points"]
        elif all(name in payload for name in
                 ("timestamp", "position_mm", "velocity_mm_s", "acceleration_mm_s2")):
            return WorldFlight(
                timestamp=float(payload["timestamp"]),
                position_mm=_xyz(payload["position_mm"], "trajectory position"),
                velocity_mm_s=_xyz(payload["velocity_mm_s"], "trajectory velocity"),
                acceleration_mm_s2=_xyz(payload["acceleration_mm_s2"], "trajectory acceleration"),
                rms_error_mm=float(payload.get("rms_error_mm", 0.0)),
            )
        else:
            raise ValueError(
                "Trajectory object must contain 'points' or a fitted world-flight model"
            )
    if not isinstance(payload, list):
        raise ValueError("Trajectory JSON must be a point array or an object containing 'points'")
    return validate_trajectory(payload)


def _check_workspace(point: Sequence[float], workspace_mm: Sequence[Sequence[float]] | None) -> None:
    if workspace_mm is None:
        return
    if len(workspace_mm) != 2:
        raise ValueError("workspace_mm must contain lower and upper XYZ bounds")
    lower = _xyz(workspace_mm[0], "workspace lower bound")
    upper = _xyz(workspace_mm[1], "workspace upper bound")
    if any(lo >= hi for lo, hi in zip(lower, upper)):
        raise ValueError("workspace lower bounds must be less than upper bounds")
    if any(value < lo or value > hi for value, lo, hi in zip(point, lower, upper)):
        raise ValueError(f"Closest trajectory point {list(point)} is outside workspace_mm")


async def move_to_trajectory(
    machine,
    trajectory: Iterable[Iterable[float]] | WorldFlight,
    *,
    arm_name: str = "arm",
    motion_name: str = "builtin",
    world_frame: str = "world",
    workspace_mm: Sequence[Sequence[float]] | None = None,
    move_timeout_s: float = 20.0,
    arrival_tolerance_mm: float = 10.0,
    trajectory_horizon_s: float = 0.5,
    execute: bool = False,
) -> dict:
    """Read the flange pose, find the closest trajectory point, and optionally move.

    The current flange orientation is preserved. Execution uses one unconstrained
    Motion service request, allowing Viam to plan against configured kinematics
    and obstacles. The configured arm driver determines its actual maximum speed.
    """
    if not arm_name or not motion_name or not world_frame:
        raise ValueError("arm, motion, and world frame names must not be empty")
    if not math.isfinite(move_timeout_s) or move_timeout_s <= 0:
        raise ValueError("move_timeout_s must be positive and finite")
    if not math.isfinite(arrival_tolerance_mm) or arrival_tolerance_mm <= 0:
        raise ValueError("arrival_tolerance_mm must be positive and finite")

    arm = Arm.from_robot(machine, arm_name)
    motion = MotionClient.from_robot(machine, motion_name)
    LOGGER.info("Reading %s end-effector pose in frame %s", arm_name, world_frame)
    current = await motion.get_pose(
        component_name=arm_name,
        destination_frame=world_frame,
        timeout=move_timeout_s,
    )
    if current.reference_frame != world_frame:
        raise ValueError(
            f"Viam returned end-effector pose in {current.reference_frame!r}, expected {world_frame!r}"
        )
    current_xyz = _xyz((current.pose.x, current.pose.y, current.pose.z), "Viam end-effector pose")
    if isinstance(trajectory, WorldFlight):
        nearest = closest_point_on_world_flight(
            current_xyz, trajectory, horizon_s=trajectory_horizon_s
        )
        trajectory_type = "world_flight"
        trajectory_points = None
    else:
        points = validate_trajectory(trajectory)
        nearest = closest_point_on_trajectory(current_xyz, points)
        trajectory_type = "polyline"
        trajectory_points = len(points)
    _check_workspace(nearest.xyz, workspace_mm)

    result = {
        "success": True,
        "executed": False,
        "reference_frame": world_frame,
        "trajectory_type": trajectory_type,
        "trajectory_points": trajectory_points,
        "start_world_mm": list(current_xyz),
        "closest": asdict(nearest),
    }
    LOGGER.info(
        "Closest %s trajectory point: %s mm (distance %.2f mm, segment=%s, fraction=%s, time=%s)",
        trajectory_type,
        [round(value, 3) for value in nearest.xyz],
        nearest.distance_mm,
        nearest.segment_index,
        nearest.segment_fraction,
        nearest.trajectory_time_s,
    )
    if not execute:
        LOGGER.info("Preview only; no motion command sent")
        return result
    if workspace_mm is None:
        raise ValueError("--execute requires explicit workspace_mm safety bounds")

    pose = current.pose
    destination = PoseInFrame(
        reference_frame=world_frame,
        pose=Pose(
            x=nearest.xyz[0],
            y=nearest.xyz[1],
            z=nearest.xyz[2],
            o_x=pose.o_x,
            o_y=pose.o_y,
            o_z=pose.o_z,
            theta=pose.theta,
        ),
    )
    commanded = False
    started = time.monotonic()
    try:
        # There is deliberately one move and no linear/orientation constraint or
        # artificial delay. Viam and the arm driver retain their configured speed
        # and safety limits; the Python Motion API has no per-call "maximum" knob.
        LOGGER.warning(
            "EXECUTE: commanding one unconstrained planned move to %s mm; "
            "physical speed is controlled by the arm configuration",
            [round(value, 3) for value in nearest.xyz],
        )
        commanded = True
        moved = await asyncio.wait_for(
            motion.move(
                component_name=arm_name,
                destination=destination,
                timeout=move_timeout_s,
            ),
            move_timeout_s,
        )
        if not moved:
            raise RuntimeError("Viam Motion service did not complete the move")

        final = await motion.get_pose(
            component_name=arm_name,
            destination_frame=world_frame,
            timeout=move_timeout_s,
        )
        if final.reference_frame != world_frame:
            raise ValueError("Viam returned the final end-effector pose in an unexpected frame")
        final_xyz = _xyz((final.pose.x, final.pose.y, final.pose.z), "final end-effector pose")
        error_mm = math.dist(final_xyz, nearest.xyz)
        if error_mm > arrival_tolerance_mm:
            raise RuntimeError(
                f"Arm stopped {error_mm:.2f} mm from the closest trajectory point "
                f"(limit {arrival_tolerance_mm:.2f} mm)"
            )
        elapsed = time.monotonic() - started
        LOGGER.info("Move complete in %.3f s; arrival error %.2f mm", elapsed, error_mm)
        result.update(
            executed=True,
            final_world_mm=list(final_xyz),
            arrival_error_mm=error_mm,
            elapsed_s=elapsed,
        )
        return result
    except BaseException:
        if commanded:
            LOGGER.exception("Move failed; requesting arm stop")
            try:
                await asyncio.shield(asyncio.wait_for(arm.stop(timeout=3), 3))
            except Exception as stop_error:
                LOGGER.error("Arm stop request failed: %s", stop_error)
        raise


def configure_logging(log_file: str | Path | None = None, verbose: bool = False) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )


async def _main(args: argparse.Namespace) -> dict:
    trajectory = load_trajectory(args.trajectory, args.world_frame)
    workspace = None
    if args.workspace:
        workspace = json.loads(args.workspace)
    async with await connect() as machine:
        return await move_to_trajectory(
            machine,
            trajectory,
            arm_name=args.arm,
            motion_name=args.motion,
            world_frame=args.world_frame,
            workspace_mm=workspace,
            move_timeout_s=args.move_timeout,
            arrival_tolerance_mm=args.arrival_tolerance,
            trajectory_horizon_s=args.trajectory_horizon,
            execute=args.execute,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True, help="JSON trajectory in world millimeters")
    parser.add_argument("--arm", default="arm", help="Viam arm/end-effector component name")
    parser.add_argument("--motion", default="builtin", help="Viam Motion service name")
    parser.add_argument("--world-frame", default="world")
    parser.add_argument(
        "--workspace",
        help='Optional JSON bounds, for example "[[0,-1000,0],[600,100,700]]"',
    )
    parser.add_argument("--move-timeout", type=float, default=20.0)
    parser.add_argument("--arrival-tolerance", type=float, default=10.0)
    parser.add_argument(
        "--trajectory-horizon", type=float, default=0.5,
        help="Seconds of a fitted WorldFlight to search (default: 0.5)",
    )
    parser.add_argument("--execute", action="store_true", help="Command one physical arm move")
    parser.add_argument("--log-file", type=Path, help="Also append logs to this file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.log_file, args.verbose)
    try:
        result = asyncio.run(_main(args))
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, RuntimeError, TimeoutError, OSError) as error:
        LOGGER.error("Move-to-trajectory failed: %s", error)
        parser.exit(2, f"Move-to-trajectory failed: {error}\n")


if __name__ == "__main__":
    main()
