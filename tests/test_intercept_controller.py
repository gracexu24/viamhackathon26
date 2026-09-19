import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from motion.intercept_controller import (
    InterceptConfig, calculate_catch_target, execute_intercept,
    image_error_to_plane_offset, validate_prediction,
)


def config(**overrides):
    values = dict(
        arm="arm", horizontal_axis="x", vertical_axis="z", plane_normal_axis="y",
        fixed_plane_value_mm=500.0,
        catch_ready_pose={"x": 100, "y": 500, "z": 200, "o_x": 0,
                          "o_y": 0, "o_z": -1, "theta": 180},
        horizontal_mm_per_pixel=2.0, vertical_mm_per_pixel=3.0,
        invert_horizontal=False, invert_vertical=False,
        horizontal_deadband_px=10.0, vertical_deadband_px=8.0,
        max_horizontal_move_mm=80.0, max_vertical_move_mm=60.0,
        horizontal_min_mm=20.0, horizontal_max_mm=180.0,
        vertical_min_mm=100.0, vertical_max_mm=300.0,
        minimum_lead_time_s=0.5, max_prediction_age_s=0.15,
        max_side_fit_error_px=8.0, max_front_fit_error_px=8.0,
        rpc_timeout_s=1.0,
    )
    values.update(overrides)
    return InterceptConfig(**values)


def prediction(horizontal=20.0, vertical=15.0, *, predicted_at=100.0, catch=101.0):
    return {
        "valid": True,
        "prediction_timestamp": predicted_at,
        "catch_timestamp": catch,
        "time_to_catch_s": catch - predicted_at,
        "predicted_front_u_at_catch": 420.0,
        "predicted_front_v_at_catch": 315.0,
        "basket_u_px": 400.0,
        "basket_v_px": 300.0,
        "horizontal_error_px": horizontal,
        "vertical_error_px": vertical,
        "side_fit_rms_px": 2.0,
        "front_fit_rms_px": 3.0,
    }


class MappingTests(unittest.TestCase):
    def test_horizontal_error_maps_to_configured_axis(self):
        target = calculate_catch_target(prediction(horizontal=20, vertical=0), config())
        self.assertEqual(target.target_pose.x, 140)
        self.assertEqual(target.target_pose.z, 200)

    def test_vertical_error_maps_to_second_axis(self):
        target = calculate_catch_target(prediction(horizontal=0, vertical=15), config())
        self.assertEqual(target.target_pose.x, 100)
        self.assertEqual(target.target_pose.z, 245)

    def test_horizontal_direction_inversion(self):
        offset = image_error_to_plane_offset(
            20, 15, config(invert_horizontal=True, invert_vertical=False))
        self.assertEqual((offset.horizontal_move_mm, offset.vertical_move_mm), (-40, 45))

    def test_vertical_direction_inversion(self):
        offset = image_error_to_plane_offset(
            20, 15, config(invert_horizontal=False, invert_vertical=True))
        self.assertEqual((offset.horizontal_move_mm, offset.vertical_move_mm), (40, -45))

    def test_horizontal_deadband_only_zeroes_horizontal(self):
        offset = image_error_to_plane_offset(10, 15, config())
        self.assertEqual(offset.horizontal_move_mm, 0)
        self.assertEqual(offset.vertical_move_mm, 45)

    def test_vertical_deadband_only_zeroes_vertical(self):
        offset = image_error_to_plane_offset(20, 8, config())
        self.assertEqual(offset.horizontal_move_mm, 40)
        self.assertEqual(offset.vertical_move_mm, 0)

    def test_separate_pixel_to_mm_scales(self):
        offset = image_error_to_plane_offset(12, 9, config())
        self.assertEqual((offset.horizontal_move_mm, offset.vertical_move_mm), (24, 27))

    def test_maximum_movements_are_clamped(self):
        offset = image_error_to_plane_offset(100, -100, config())
        self.assertEqual((offset.horizontal_move_mm, offset.vertical_move_mm), (80, -60))
        self.assertTrue(offset.horizontal_clamped)
        self.assertTrue(offset.vertical_clamped)

    def test_required_clamp_is_rejected_for_live_command(self):
        with self.assertRaisesRegex(ValueError, "exceeds_correction_limit"):
            validate_prediction(prediction(horizontal=100), config(), now_s=100.05)

    def test_catch_plane_bounds_are_enforced(self):
        unsafe = config(catch_ready_pose={"x": 170, "y": 500, "z": 200,
                                          "o_x": 0, "o_y": 0, "o_z": -1,
                                          "theta": 180})
        with self.assertRaisesRegex(ValueError, "horizontal_workspace"):
            calculate_catch_target(prediction(horizontal=20, vertical=0), unsafe)

    def test_plane_normal_coordinate_remains_fixed(self):
        target = calculate_catch_target(prediction(), config())
        self.assertEqual(target.target_pose.y, 500)

    def test_basket_orientation_remains_fixed(self):
        target = calculate_catch_target(prediction(), config())
        pose = target.target_pose
        self.assertEqual((pose.o_x, pose.o_y, pose.o_z, pose.theta), (0, 0, -1, 180))

    def test_insufficient_lead_time_rejected(self):
        with self.assertRaisesRegex(ValueError, "insufficient_lead_time"):
            validate_prediction(prediction(predicted_at=100, catch=100.4), config(), now_s=100.05)

    def test_stale_prediction_rejected(self):
        with self.assertRaisesRegex(ValueError, "stale_prediction"):
            validate_prediction(prediction(predicted_at=99.0, catch=101), config(), now_s=100.0)

    def test_nonfinite_error_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_prediction(prediction(horizontal=float("nan")), config(), now_s=100.05)

    def test_bad_fit_quality_rejected(self):
        value = prediction()
        value["front_fit_rms_px"] = 9.0
        with self.assertRaisesRegex(ValueError, "fit_quality"):
            validate_prediction(value, config(), now_s=100.05)


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_dry_run_never_gets_arm(self):
        with patch("motion.intercept_controller.Arm.from_robot") as factory:
            result = await execute_intercept(None, prediction(), config(), now_s=100.05)
        self.assertEqual(result["action"], "WOULD_MOVE")
        factory.assert_not_called()

    async def test_valid_execution_sends_exactly_one_move(self):
        arm = SimpleNamespace(is_moving=AsyncMock(return_value=False),
                              move_to_position=AsyncMock(), stop=AsyncMock())
        with patch("motion.intercept_controller.Arm.from_robot", return_value=arm):
            result = await execute_intercept(object(), prediction(), config(),
                                             execute=True, now_s=100.05)
        self.assertEqual(result["action"], "MOVED_AND_HOLDING")
        arm.move_to_position.assert_awaited_once()
        arm.stop.assert_not_awaited()

    async def test_failed_motion_is_not_retried(self):
        arm = SimpleNamespace(is_moving=AsyncMock(return_value=False),
                              move_to_position=AsyncMock(side_effect=RuntimeError("boom")),
                              stop=AsyncMock())
        with patch("motion.intercept_controller.Arm.from_robot", return_value=arm):
            result = await execute_intercept(object(), prediction(), config(),
                                             execute=True, now_s=100.05)
        self.assertEqual(result["action"], "FAILED")
        arm.move_to_position.assert_awaited_once()
        arm.stop.assert_awaited_once()

    async def test_already_moving_arm_aborts_without_command(self):
        arm = SimpleNamespace(is_moving=AsyncMock(return_value=True),
                              move_to_position=AsyncMock(), stop=AsyncMock())
        with patch("motion.intercept_controller.Arm.from_robot", return_value=arm):
            result = await execute_intercept(object(), prediction(), config(),
                                             execute=True, now_s=100.05)
        self.assertEqual(result["reason"], "arm_already_moving")
        arm.move_to_position.assert_not_awaited()
        arm.stop.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
