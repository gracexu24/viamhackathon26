import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from motion.lateral_catch import (
    LateralCatchConfig, build_lateral_target, run_lateral_catch,
    validate_config, validate_prediction,
)


def config(**overrides):
    values = dict(
        arm="arm", lateral_axis="y",
        base_pose={"x": 10, "y": 100, "z": 500, "o_x": 0, "o_y": 0,
                   "o_z": -1, "theta": 180},
        mm_per_pixel=2.0, max_lateral_move_mm=80.0,
        lateral_min_mm=20.0, lateral_max_mm=180.0,
        max_prediction_age_s=.15,
        center_deadband_px=10.0, robot_sign_for_positive_error=1.0,
        rpc_timeout_s=1.0,
    )
    values.update(overrides)
    return LateralCatchConfig(**values)


def prediction(error=20, decision="RIGHT", catch=101.0, now=100.0):
    return {
        "catch_timestamp": catch, "time_to_catch_s": catch-now,
        "predicted_ball_u_at_catch": 420, "basket_u_px": 400,
        "lateral_error_px": error, "decision": decision,
        "side_fit_rms_px": 2, "front_fit_rms_px": 2,
    }


class LateralMappingTests(unittest.TestCase):
    def test_positive_error_maps_to_configured_physical_direction(self):
        target = build_lateral_target(prediction(20), config())
        self.assertEqual(target.axis, "y")
        self.assertEqual(target.move_mm, 40)
        self.assertEqual(target.target_lateral_mm, 140)

    def test_negative_error_maps_opposite_direction(self):
        target = build_lateral_target(prediction(-20, "LEFT"), config())
        self.assertEqual(target.move_mm, -40)

    def test_center_produces_zero_movement(self):
        target = build_lateral_target(prediction(0, "CENTER"), config())
        self.assertEqual(target.move_mm, 0)

    def test_mm_per_pixel_conversion_and_clamp(self):
        target = build_lateral_target(prediction(100, "RIGHT"), config())
        self.assertEqual(target.move_mm, 80)

    def test_unsafe_final_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside safe bounds"):
            build_lateral_target(prediction(-20, "LEFT"), config(base_pose={
                "x": 10, "y": 30, "z": 500, "o_x": 0, "o_y": 0,
                "o_z": -1, "theta": 180}))

    def test_prediction_validation_accepts_short_lead(self):
        _, catch_in, _ = validate_prediction(
            prediction(catch=100.2), now_s=100, config=config())
        self.assertAlmostEqual(catch_in, 0.2)

    def test_prediction_validation_rejects_stale(self):
        with self.assertRaisesRegex(ValueError, "stale"):
            validate_prediction(prediction(catch=99), now_s=100, config=config())


class ArmExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_dry_run_never_calls_arm(self):
        with patch("motion.lateral_catch.Arm.from_robot") as factory:
            result = await run_lateral_catch(None, prediction(), config(), now_s=100)
        self.assertEqual(result["action"], "WOULD_MOVE")
        factory.assert_not_called()

    async def test_valid_execution_calls_arm_once(self):
        arm = SimpleNamespace(is_moving=AsyncMock(return_value=False),
                              move_to_position=AsyncMock(), stop=AsyncMock())
        with patch("motion.lateral_catch.Arm.from_robot", return_value=arm):
            result = await run_lateral_catch(object(), prediction(), config(),
                                             execute=True, now_s=100)
        self.assertTrue(result["executed"])
        arm.move_to_position.assert_awaited_once()
        arm.stop.assert_not_awaited()

    async def test_motion_preserves_fixed_height_and_orientation(self):
        arm = SimpleNamespace(is_moving=AsyncMock(return_value=False),
                              move_to_position=AsyncMock(), stop=AsyncMock())
        with patch("motion.lateral_catch.Arm.from_robot", return_value=arm):
            await run_lateral_catch(object(), prediction(), config(), execute=True, now_s=100)
        pose = arm.move_to_position.await_args.kwargs["pose"]
        self.assertEqual((pose.x, pose.z, pose.o_x, pose.o_y, pose.o_z, pose.theta),
                         (10, 500, 0, 0, -1, 180))

    async def test_center_never_moves(self):
        arm = SimpleNamespace(is_moving=AsyncMock(return_value=False),
                              move_to_position=AsyncMock(), stop=AsyncMock())
        with patch("motion.lateral_catch.Arm.from_robot", return_value=arm):
            result = await run_lateral_catch(object(), prediction(5, "CENTER"), config(),
                                             execute=True, now_s=100)
        self.assertEqual(result["action"], "HOLD_CENTER")
        arm.move_to_position.assert_not_awaited()

    async def test_failed_motion_stops_and_does_not_retry(self):
        arm = SimpleNamespace(is_moving=AsyncMock(return_value=False),
                              move_to_position=AsyncMock(side_effect=RuntimeError("boom")),
                              stop=AsyncMock())
        with patch("motion.lateral_catch.Arm.from_robot", return_value=arm):
            result = await run_lateral_catch(object(), prediction(), config(),
                                             execute=True, now_s=100)
        self.assertFalse(result["success"])
        arm.move_to_position.assert_awaited_once()
        arm.stop.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
