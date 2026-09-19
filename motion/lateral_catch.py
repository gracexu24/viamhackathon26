"""One-shot lateral basket move driven by a Phase 2 prediction.

The default is a dry run. This module contains no camera processing; callers
pass a CatchPrediction (or an equivalent JSON mapping) from Phase 2.
"""

import argparse
import asyncio
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from viam.components.arm import Arm
from viam.proto.common import Pose

from connection import connect


@dataclass(frozen=True)
class LateralCatchConfig:
    arm: str
    lateral_axis: str
    base_pose: dict
    mm_per_pixel: float
    max_lateral_move_mm: float
    lateral_min_mm: float
    lateral_max_mm: float
    max_prediction_age_s: float = 0.15
    center_deadband_px: float = 25.0
    robot_sign_for_positive_error: float = 1.0
    rpc_timeout_s: float = 5.0


@dataclass(frozen=True)
class LateralTarget:
    axis: str
    error_px: float
    move_mm: float
    target_lateral_mm: float
    target_pose: Pose


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def validate_config(config):
    if not isinstance(config.arm, str) or not config.arm.strip():
        raise ValueError("arm must be a nonempty resource name")
    if config.lateral_axis not in ("x", "y"):
        raise ValueError("lateral_axis must be 'x' or 'y'")
    if not isinstance(config.base_pose, dict):
        raise ValueError("base_pose must be an object containing a tested catch-ready pose")
    for key in ("x", "y", "z", "o_x", "o_y", "o_z", "theta"):
        _finite(config.base_pose.get(key), f"base_pose.{key}")
    for key in ("mm_per_pixel", "max_lateral_move_mm", "max_prediction_age_s",
                "rpc_timeout_s"):
        if _finite(getattr(config, key), key) <= 0:
            raise ValueError(f"{key} must be positive")
    lower = _finite(config.lateral_min_mm, "lateral_min_mm")
    upper = _finite(config.lateral_max_mm, "lateral_max_mm")
    if lower >= upper:
        raise ValueError("lateral_min_mm must be less than lateral_max_mm")
    sign = _finite(config.robot_sign_for_positive_error, "robot_sign_for_positive_error")
    if sign not in (-1.0, 1.0):
        raise ValueError("robot_sign_for_positive_error must be +1 or -1")
    if _finite(config.center_deadband_px, "center_deadband_px") < 0:
        raise ValueError("center_deadband_px must be nonnegative")


def prediction_value(prediction, name):
    if isinstance(prediction, dict):
        value = prediction.get(name)
    else:
        value = getattr(prediction, name, None)
    return _finite(value, f"prediction.{name}")


def validate_prediction(prediction, *, now_s=None, config):
    validate_config(config)
    now_s = time.time() if now_s is None else _finite(now_s, "now_s")
    catch_timestamp = prediction_value(prediction, "catch_timestamp")
    time_to_catch = prediction_value(prediction, "time_to_catch_s")
    error_px = prediction_value(prediction, "lateral_error_px")
    decision = prediction.get("decision") if isinstance(prediction, dict) else getattr(prediction, "decision", None)
    if decision not in ("LEFT", "CENTER", "RIGHT"):
        raise ValueError("prediction.decision must be LEFT, CENTER, or RIGHT")
    for name in ("predicted_ball_u_at_catch", "basket_u_px", "side_fit_rms_px", "front_fit_rms_px"):
        prediction_value(prediction, name)
    if time_to_catch <= 0 or catch_timestamp <= now_s:
        raise ValueError("prediction is stale or catch time is not in the future")
    if abs((catch_timestamp - now_s) - time_to_catch) > max(config.max_prediction_age_s, 0.05):
        raise ValueError("prediction timing is stale or inconsistent")
    if abs(error_px) > config.max_lateral_move_mm / config.mm_per_pixel:
        raise ValueError("lateral pixel error exceeds configured movement limit")
    return catch_timestamp, catch_timestamp - now_s, error_px


def build_lateral_target(prediction, config):
    """Convert pixel error into one bounded Pose, changing only x or y."""
    validate_config(config)
    error_px = prediction_value(prediction, "lateral_error_px")
    move_mm = error_px * config.mm_per_pixel * config.robot_sign_for_positive_error
    move_mm = max(-config.max_lateral_move_mm, min(config.max_lateral_move_mm, move_mm))
    base_lateral = _finite(config.base_pose[config.lateral_axis],
                           f"base_pose.{config.lateral_axis}")
    target_lateral = base_lateral + move_mm
    if not config.lateral_min_mm <= target_lateral <= config.lateral_max_mm:
        raise ValueError("final lateral target is outside safe bounds")
    values = dict(config.base_pose)
    values[config.lateral_axis] = target_lateral
    return LateralTarget(config.lateral_axis, error_px, move_mm, target_lateral,
                         Pose(**values))


def target_summary(target):
    pose = target.target_pose
    return {"axis": target.axis, "error_px": target.error_px, "move_mm": target.move_mm,
            "target_lateral_mm": target.target_lateral_mm,
            "target_pose": {"x": pose.x, "y": pose.y, "z": pose.z,
                            "o_x": pose.o_x, "o_y": pose.o_y, "o_z": pose.o_z,
                            "theta": pose.theta}}


async def run_lateral_catch(machine, prediction, config, *, execute=False, now_s=None):
    """Validate one prediction and optionally issue exactly one arm move."""
    catch_timestamp, catch_in, error_px = validate_prediction(
        prediction, now_s=now_s, config=config)
    target = build_lateral_target(prediction, config)
    decision = prediction.get("decision") if isinstance(prediction, dict) else getattr(prediction, "decision", None)
    if abs(error_px) <= config.center_deadband_px or decision == "CENTER":
        return {"success": True, "executed": False, "action": "HOLD_CENTER",
                "decision": "CENTER", "catch_in": catch_in, "target": target_summary(target)}
    result = {"success": True, "executed": False, "action": "WOULD_MOVE",
              "decision": decision, "catch_in": catch_in, "target": target_summary(target)}
    if not execute:
        return result

    arm = Arm.from_robot(machine, config.arm)
    commanded = False
    try:
        if await arm.is_moving(timeout=config.rpc_timeout_s):
            raise ValueError("arm is already moving")
        commanded = True
        await arm.move_to_position(pose=target.target_pose)
        result.update(executed=True, action="MOVED_AND_HOLDING")
        return result
    except asyncio.CancelledError:
        if commanded:
            await asyncio.shield(arm.stop(timeout=config.rpc_timeout_s))
        raise
    except Exception as error:
        if commanded:
            try:
                await arm.stop(timeout=config.rpc_timeout_s)
            except Exception as stop_error:
                result["stop_error"] = str(stop_error)
        result.update(success=False, executed=commanded, action="FAILED",
                      reason=f"{type(error).__name__}: {error}")
        return result


def config_from_mapping(raw):
    # Accept old config files while intentionally ignoring the retired timing gate.
    raw = dict(raw)
    raw.pop("minimum_lead_time_s", None)
    return LateralCatchConfig(**raw)


async def _main(args):
    config = config_from_mapping(json.loads(Path(args.config).read_text(encoding="utf-8")))
    prediction = json.loads(Path(args.prediction_json).read_text(encoding="utf-8"))
    if args.execute:
        async with await connect() as machine:
            result = await run_lateral_catch(machine, prediction, config, execute=True)
    else:
        result = await run_lateral_catch(None, prediction, config, execute=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Measured basket-ready pose and safety config JSON")
    parser.add_argument("--prediction-json", required=True, help="Phase 2 CatchPrediction JSON")
    parser.add_argument("--execute", action="store_true", help="Actually move the arm; default is dry-run")
    args = parser.parse_args()
    try:
        result = asyncio.run(_main(args))
        print(json.dumps(result, indent=2, allow_nan=False, default=lambda value: value.__dict__))
        if not result["success"]:
            raise SystemExit(2)
    except (ValueError, OSError, RuntimeError, TimeoutError) as error:
        parser.exit(2, f"Lateral catch failed: {error}\n")


if __name__ == "__main__":
    main()
