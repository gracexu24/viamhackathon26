"""One-shot two-axis UF850 basket intercept in a fixed catch plane.

Perception remains in ``motion.two_camera_rgb_local``.  This module consumes
one structured Phase 2 prediction, maps front-image (u, v) error to two
configured Cartesian axes, and defaults to a hardware-safe dry run.
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
from viam.robot.client import RobotClient

from motion import two_camera_rgb_local as phase2
from vision.local_camera import credentials, positive


POSITION_AXES = ("x", "y", "z")
POSE_FIELDS = ("x", "y", "z", "o_x", "o_y", "o_z", "theta")


@dataclass(frozen=True)
class InterceptConfig:
    arm: str
    horizontal_axis: str
    vertical_axis: str
    plane_normal_axis: str
    fixed_plane_value_mm: float
    catch_ready_pose: dict
    horizontal_mm_per_pixel: float
    vertical_mm_per_pixel: float
    invert_horizontal: bool
    invert_vertical: bool
    horizontal_deadband_px: float
    vertical_deadband_px: float
    max_horizontal_move_mm: float
    max_vertical_move_mm: float
    horizontal_min_mm: float
    horizontal_max_mm: float
    vertical_min_mm: float
    vertical_max_mm: float
    minimum_lead_time_s: float
    max_prediction_age_s: float
    max_side_fit_error_px: float
    max_front_fit_error_px: float
    rpc_timeout_s: float = 5.0


@dataclass(frozen=True)
class PlaneOffset:
    horizontal_error_px: float
    vertical_error_px: float
    horizontal_move_mm: float
    vertical_move_mm: float
    horizontal_raw_mm: float
    vertical_raw_mm: float
    horizontal_clamped: bool
    vertical_clamped: bool


@dataclass(frozen=True)
class CatchTarget:
    offset: PlaneOffset
    target_pose: Pose


def _finite(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _prediction_value(prediction, name):
    value = prediction.get(name) if isinstance(prediction, dict) else getattr(prediction, name, None)
    return _finite(value, f"prediction.{name}")


def validate_config(config):
    if not isinstance(config.arm, str) or not config.arm.strip():
        raise ValueError("arm must be a nonempty resource name")
    axes = (config.horizontal_axis, config.vertical_axis, config.plane_normal_axis)
    if any(axis not in POSITION_AXES for axis in axes) or len(set(axes)) != 3:
        raise ValueError("horizontal, vertical, and plane-normal axes must be distinct x/y/z axes")
    if not isinstance(config.catch_ready_pose, dict):
        raise ValueError("catch_ready_pose must be a physically verified pose object")
    for field in POSE_FIELDS:
        _finite(config.catch_ready_pose.get(field), f"catch_ready_pose.{field}")
    fixed_plane = _finite(config.fixed_plane_value_mm, "fixed_plane_value_mm")
    for name in ("horizontal_mm_per_pixel", "vertical_mm_per_pixel",
                 "max_horizontal_move_mm", "max_vertical_move_mm",
                 "minimum_lead_time_s", "max_prediction_age_s",
                 "max_side_fit_error_px", "max_front_fit_error_px", "rpc_timeout_s"):
        if _finite(getattr(config, name), name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("horizontal_deadband_px", "vertical_deadband_px"):
        if _finite(getattr(config, name), name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if not isinstance(config.invert_horizontal, bool) or not isinstance(config.invert_vertical, bool):
        raise ValueError("invert_horizontal and invert_vertical must be booleans")
    if _finite(config.horizontal_min_mm, "horizontal_min_mm") >= _finite(
            config.horizontal_max_mm, "horizontal_max_mm"):
        raise ValueError("horizontal_min_mm must be less than horizontal_max_mm")
    if _finite(config.vertical_min_mm, "vertical_min_mm") >= _finite(
            config.vertical_max_mm, "vertical_max_mm"):
        raise ValueError("vertical_min_mm must be less than vertical_max_mm")
    if not math.isclose(float(config.catch_ready_pose[config.plane_normal_axis]),
                        fixed_plane, abs_tol=1e-9):
        raise ValueError("catch_ready_pose must lie on fixed_plane_value_mm")
    if not config.horizontal_min_mm <= float(
            config.catch_ready_pose[config.horizontal_axis]) <= config.horizontal_max_mm:
        raise ValueError("catch_ready_pose is outside horizontal workspace")
    if not config.vertical_min_mm <= float(
            config.catch_ready_pose[config.vertical_axis]) <= config.vertical_max_mm:
        raise ValueError("catch_ready_pose is outside vertical workspace")


def image_error_to_plane_offset(horizontal_error_px, vertical_error_px, config):
    """Map image errors independently, applying deadbands, signs, and clamps."""
    validate_config(config)
    horizontal_error = _finite(horizontal_error_px, "horizontal_error_px")
    vertical_error = _finite(vertical_error_px, "vertical_error_px")
    active_h = 0.0 if abs(horizontal_error) <= config.horizontal_deadband_px else horizontal_error
    active_v = 0.0 if abs(vertical_error) <= config.vertical_deadband_px else vertical_error
    horizontal_raw = active_h * config.horizontal_mm_per_pixel * (-1 if config.invert_horizontal else 1)
    vertical_raw = active_v * config.vertical_mm_per_pixel * (-1 if config.invert_vertical else 1)
    horizontal_move = max(-config.max_horizontal_move_mm,
                          min(config.max_horizontal_move_mm, horizontal_raw))
    vertical_move = max(-config.max_vertical_move_mm,
                        min(config.max_vertical_move_mm, vertical_raw))
    return PlaneOffset(
        horizontal_error, vertical_error, horizontal_move, vertical_move,
        horizontal_raw, vertical_raw,
        not math.isclose(horizontal_move, horizontal_raw),
        not math.isclose(vertical_move, vertical_raw),
    )


def validate_prediction(prediction, config, *, now_s=None):
    """Reject an invalid/stale/late Phase 2 result before target generation."""
    validate_config(config)
    valid = prediction.get("valid") if isinstance(prediction, dict) else getattr(prediction, "valid", None)
    if valid is not True:
        raise ValueError("phase2_prediction_invalid")
    now_s = time.time() if now_s is None else _finite(now_s, "now_s")
    predicted_at = _prediction_value(prediction, "prediction_timestamp")
    catch_timestamp = _prediction_value(prediction, "catch_timestamp")
    reported_time_to_catch = _prediction_value(prediction, "time_to_catch_s")
    horizontal_error = _prediction_value(prediction, "horizontal_error_px")
    vertical_error = _prediction_value(prediction, "vertical_error_px")
    side_rms = _prediction_value(prediction, "side_fit_rms_px")
    front_rms = _prediction_value(prediction, "front_fit_rms_px")
    for name in ("predicted_front_u_at_catch", "predicted_front_v_at_catch",
                 "basket_u_px", "basket_v_px"):
        _prediction_value(prediction, name)
    if catch_timestamp <= now_s or reported_time_to_catch <= 0:
        raise ValueError("catch_is_in_the_past")
    if now_s - predicted_at > config.max_prediction_age_s or predicted_at > now_s + 0.05:
        raise ValueError("stale_prediction")
    if abs((catch_timestamp - predicted_at) - reported_time_to_catch) > 0.05:
        raise ValueError("inconsistent_prediction_timing")
    catch_in = catch_timestamp - now_s
    if catch_in <= config.minimum_lead_time_s:
        raise ValueError("insufficient_lead_time")
    if side_rms > config.max_side_fit_error_px or front_rms > config.max_front_fit_error_px:
        raise ValueError("fit_quality_outside_limits")
    offset = image_error_to_plane_offset(horizontal_error, vertical_error, config)
    if offset.horizontal_clamped or offset.vertical_clamped:
        raise ValueError("required_movement_exceeds_correction_limit")
    return catch_in, offset


def calculate_catch_target(prediction, config):
    """Build a pose from the verified base pose and two in-plane offsets."""
    offset = image_error_to_plane_offset(
        _prediction_value(prediction, "horizontal_error_px"),
        _prediction_value(prediction, "vertical_error_px"), config)
    values = {field: _finite(config.catch_ready_pose.get(field),
                             f"catch_ready_pose.{field}") for field in POSE_FIELDS}
    values[config.horizontal_axis] += offset.horizontal_move_mm
    values[config.vertical_axis] += offset.vertical_move_mm
    values[config.plane_normal_axis] = _finite(
        config.fixed_plane_value_mm, "fixed_plane_value_mm")
    target = CatchTarget(offset=offset, target_pose=Pose(**values))
    validate_catch_target(target, config)
    return target


def validate_catch_target(target, config):
    pose = target.target_pose
    horizontal = _finite(getattr(pose, config.horizontal_axis), "target horizontal axis")
    vertical = _finite(getattr(pose, config.vertical_axis), "target vertical axis")
    normal = _finite(getattr(pose, config.plane_normal_axis), "target plane-normal axis")
    if not config.horizontal_min_mm <= horizontal <= config.horizontal_max_mm:
        raise ValueError("target_outside_horizontal_workspace")
    if not config.vertical_min_mm <= vertical <= config.vertical_max_mm:
        raise ValueError("target_outside_vertical_workspace")
    if not math.isclose(normal, config.fixed_plane_value_mm, abs_tol=1e-9):
        raise ValueError("target_left_fixed_catch_plane")
    for field in ("o_x", "o_y", "o_z", "theta"):
        if not math.isclose(getattr(pose, field), float(config.catch_ready_pose[field]), abs_tol=1e-9):
            raise ValueError("target_changed_basket_orientation")


def _pose_mapping(pose):
    return {field: getattr(pose, field) for field in POSE_FIELDS}


def _summary(prediction, catch_in, target):
    return {
        "catch_in_s": catch_in,
        "predicted_ball_px": [
            _prediction_value(prediction, "predicted_front_u_at_catch"),
            _prediction_value(prediction, "predicted_front_v_at_catch"),
        ],
        "basket_center_px": [
            _prediction_value(prediction, "basket_u_px"),
            _prediction_value(prediction, "basket_v_px"),
        ],
        "error_px": [target.offset.horizontal_error_px, target.offset.vertical_error_px],
        "horizontal_move_mm": target.offset.horizontal_move_mm,
        "vertical_move_mm": target.offset.vertical_move_mm,
        "target_pose": _pose_mapping(target.target_pose),
    }


async def execute_intercept(machine, prediction, config, *, execute=False, now_s=None):
    """Validate and commit at most one UF850 movement command."""
    try:
        catch_in, _ = validate_prediction(prediction, config, now_s=now_s)
        target = calculate_catch_target(prediction, config)
    except ValueError as error:
        return {"success": False, "executed": False, "action": "ABORT", "reason": str(error)}

    result = {"success": True, "executed": False, "action": "WOULD_MOVE",
              **_summary(prediction, catch_in, target)}
    centered = (target.offset.horizontal_move_mm == 0 and target.offset.vertical_move_mm == 0)
    if centered:
        result["action"] = "CENTER_HOLD"
        return result
    if not execute:
        return result

    arm = Arm.from_robot(machine, config.arm)
    try:
        if await arm.is_moving(timeout=config.rpc_timeout_s):
            return {**result, "success": False, "action": "ABORT",
                    "reason": "arm_already_moving"}
        await arm.move_to_position(pose=target.target_pose)
        result.update(executed=True, action="MOVED_AND_HOLDING")
        return result
    except asyncio.CancelledError:
        await asyncio.shield(arm.stop(timeout=config.rpc_timeout_s))
        raise
    except Exception as error:
        try:
            await arm.stop(timeout=config.rpc_timeout_s)
        except Exception as stop_error:
            result["stop_error"] = f"{type(stop_error).__name__}: {stop_error}"
        result.update(success=False, executed=True, action="FAILED",
                      reason=f"{type(error).__name__}: {error}")
        return result


def config_from_mapping(raw):
    try:
        config = InterceptConfig(**raw)
    except TypeError as error:
        raise ValueError(f"invalid intercept configuration: {error}") from error
    validate_config(config)
    return config


def _apply_cli_overrides(raw, args):
    names = (
        "arm", "horizontal_axis", "vertical_axis", "plane_normal_axis",
        "fixed_plane_value_mm", "horizontal_mm_per_pixel", "vertical_mm_per_pixel",
        "horizontal_deadband_px", "vertical_deadband_px",
        "max_horizontal_move_mm", "max_vertical_move_mm", "minimum_lead_time_s",
        "horizontal_min_mm", "horizontal_max_mm", "vertical_min_mm", "vertical_max_mm",
        "invert_horizontal", "invert_vertical",
    )
    for name in names:
        value = getattr(args, name, None)
        if value is not None:
            raw[name] = value
    return raw


async def _connect_local(machine_config):
    key_id, key = credentials(machine_config)
    options = RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id)
    options.dial_options.disable_webrtc = True
    options.dial_options.timeout = 10
    return await RobotClient.at_address("127.0.0.1:8080", options)


async def _main(args):
    raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config = config_from_mapping(_apply_cli_overrides(raw, args))
    if args.prediction_json:
        prediction = json.loads(Path(args.prediction_json).read_text(encoding="utf-8"))
        if not args.execute:
            return await execute_intercept(None, prediction, config, execute=False)
        if args.machine_config is None:
            raise ValueError("--machine-config is required with --execute")
        async with await _connect_local(args.machine_config) as machine:
            return await execute_intercept(machine, prediction, config, execute=True)
    if args.machine_config is None:
        raise ValueError("--machine-config is required for live camera prediction")

    # Phase 2 uses these names internally; Phase 3 owns the two independent
    # deadbands and does not rely on its LEFT/CENTER/RIGHT label for movement.
    args.center_deadband_px = config.horizontal_deadband_px
    args.vertical_deadband_px = config.vertical_deadband_px

    async def consume(machine, prediction):
        return await execute_intercept(machine, prediction, config, execute=args.execute)

    return await phase2.run(args, prediction_handler=consume)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Measured catch-plane calibration JSON")
    parser.add_argument("--prediction-json", help="Use one saved Phase 2 prediction instead of cameras")
    parser.add_argument("--execute", action="store_true", help="Actually move arm; default is dry-run")
    parser.add_argument("--arm")
    parser.add_argument("--horizontal-mm-per-pixel", type=positive)
    parser.add_argument("--vertical-mm-per-pixel", type=positive)
    parser.add_argument("--horizontal-deadband-px", type=float)
    parser.add_argument("--vertical-deadband-px", type=float)
    parser.add_argument("--max-horizontal-move-mm", type=positive)
    parser.add_argument("--max-vertical-move-mm", type=positive)
    parser.add_argument("--minimum-lead-time-s", type=positive)
    parser.add_argument("--invert-horizontal", action="store_true", default=None)
    parser.add_argument("--invert-vertical", action="store_true", default=None)
    parser.add_argument("--horizontal-axis", choices=POSITION_AXES)
    parser.add_argument("--vertical-axis", choices=POSITION_AXES)
    parser.add_argument("--plane-normal-axis", choices=POSITION_AXES)
    parser.add_argument("--fixed-plane-value-mm", type=float)
    parser.add_argument("--horizontal-min-mm", type=float)
    parser.add_argument("--horizontal-max-mm", type=float)
    parser.add_argument("--vertical-min-mm", type=float)
    parser.add_argument("--vertical-max-mm", type=float)

    # Live Phase 2 inputs.  They deliberately mirror two_camera_rgb_local.py.
    parser.add_argument("--machine-config", type=Path)
    parser.add_argument("--side-camera", default="cam2")
    parser.add_argument("--front-camera", default="cam")
    parser.add_argument("--color-source", default="color")
    parser.add_argument("--catch-u-px", type=float)
    parser.add_argument("--basket-u-px", type=float)
    parser.add_argument("--basket-v-px", type=float)
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
    return parser


def main():
    parser = _parser()
    args = parser.parse_args()
    if not args.prediction_json:
        missing = [name for name in ("catch_u_px", "basket_u_px", "basket_v_px")
                   if getattr(args, name) is None]
        if missing:
            parser.error("live mode requires --" + ", --".join(name.replace("_", "-") for name in missing))
    try:
        result = asyncio.run(_main(args))
        print(json.dumps(result, indent=2, allow_nan=False))
        if result is None or not result.get("success", False):
            raise SystemExit(2)
    except (ValueError, OSError, RuntimeError, TimeoutError) as error:
        parser.exit(2, f"Intercept controller failed: {error}\n")


if __name__ == "__main__":
    main()
