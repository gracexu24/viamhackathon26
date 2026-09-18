import asyncio
import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from viam.proto.common import Geometry, GeometriesInFrame, PointCloudObject, Pose, PoseInFrame

from motion.viam_stationary_grab import DEFAULT_CONFIG, _observe, make_targets, run_stationary_grab, validate_config
from vision.viam_ball import BallDetection, BallPose


def config():
    return json.loads(DEFAULT_CONFIG.read_text())


BALL = BallPose(197.6, -818.8, 17.9, "world", "ball", 100)
DETECTION = BallDetection("ball", 1, 720, 426, 801, 511)


class TargetTests(unittest.TestCase):
    def test_taught_tcp_height_does_not_descend_to_segment_height(self):
        cfg = config()
        targets = make_targets(BALL, cfg)
        self.assertEqual(targets["grasp"][:2], [197.6, -818.8])
        self.assertAlmostEqual(targets["grasp"][2], 78.57077725670194)
        self.assertEqual(targets["approach"][2], targets["grasp"][2]+100)
        self.assertEqual(targets["lift"][2], targets["grasp"][2]+125)

    def test_wrong_frame_outside_workspace_and_wrong_table_fail(self):
        for ball in (BallPose(0, 0, 17, "cam", "ball", 1),
                     BallPose(999, -818, 17, "world", "ball", 1),
                     BallPose(198, -818, 100, "world", "ball", 1),
                     BallPose(float("nan"), -818, 17, "world", "ball", 1)):
            with self.assertRaises(ValueError):
                make_targets(ball, config())

    def test_invalid_configuration_rejected(self):
        for key, value in (("move_timeout_s", float("nan")), ("settle_s", -1),
                           ("grasp_z_world_mm", float("inf")), ("grasp_orientation", [0, 0, 1, 0])):
            cfg = config()
            cfg[key] = value
            with self.assertRaises(ValueError):
                validate_config(cfg)


class CycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cfg = config()
        self.events = []
        self.pose = Pose(x=228, y=-822, z=self.cfg["grasp_z_world_mm"], o_z=-1)
        self.arm = SimpleNamespace(is_moving=AsyncMock(return_value=False), stop=AsyncMock())

        async def move(**kwargs):
            self.events.append(("move", kwargs))
            self.pose.CopyFrom(kwargs["destination"].pose)
            return True

        async def get_pose(*args, **kwargs):
            return PoseInFrame(reference_frame="world", pose=self.pose)

        async def open_gripper(**kwargs):
            self.events.append(("open", None))

        async def grab(**kwargs):
            self.events.append(("grab", None))
            return True

        self.motion = SimpleNamespace(move=AsyncMock(side_effect=move), get_pose=AsyncMock(side_effect=get_pose))
        self.gripper = SimpleNamespace(open=AsyncMock(side_effect=open_gripper),
                                       grab=AsyncMock(side_effect=grab), stop=AsyncMock())
        self.observe = AsyncMock(return_value=(BALL, DETECTION))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in (("Arm.from_robot", self.arm), ("Gripper.from_robot", self.gripper),
                            ("MotionClient.from_robot", self.motion)):
            self.stack.enter_context(patch("motion.viam_stationary_grab."+name, return_value=value))
        self.stack.enter_context(patch("motion.viam_stationary_grab._observe", self.observe))

    async def test_preview_has_no_actuator_calls(self):
        result = await run_stationary_grab(object(), self.cfg)
        self.assertTrue(result["success"])
        self.assertFalse(result["executed"])
        self.assertEqual(result["state"], "preview")
        self.assertEqual(self.events, [])
        self.arm.stop.assert_not_awaited()
        self.gripper.stop.assert_not_awaited()

    async def test_cycle_moves_tcp_in_world_and_only_lifts_after_grab(self):
        result = await run_stationary_grab(object(), self.cfg, execute=True)
        self.assertEqual(result["state"], "ball_grabbed")
        self.assertTrue(result["grabbed"])
        events = [name for name, _ in self.events]
        self.assertEqual(events, ["move", "move", "move", "open", "move", "grab", "move"])
        moves = [kwargs for name, kwargs in self.events if name == "move"]
        for move in moves:
            self.assertEqual(move["component_name"], "gripper")
            self.assertEqual(move["destination"].reference_frame, "world")
        self.assertAlmostEqual(moves[-2]["destination"].pose.z, self.cfg["grasp_z_world_mm"])
        self.assertTrue(moves[-2]["constraints"].linear_constraint)
        self.assertTrue(moves[-1]["constraints"].linear_constraint)

    async def test_approach_only_never_opens_or_grabs(self):
        result = await run_stationary_grab(object(), self.cfg, execute=True, approach_only=True)
        self.assertEqual(result["state"], "approached")
        self.assertEqual(self.motion.move.await_count, 2)
        self.gripper.open.assert_not_awaited()
        self.gripper.grab.assert_not_awaited()

    async def test_no_ball_is_read_only_failure(self):
        self.observe.side_effect = ValueError("No ball")
        result = await run_stationary_grab(object(), self.cfg, execute=True)
        self.assertFalse(result["success"])
        self.assertFalse(result["executed"])
        self.assertEqual(self.events, [])

    async def test_lost_ball_after_approach_stops_without_gripping(self):
        self.observe.side_effect = [(BALL, DETECTION), ValueError("No ball")]
        result = await run_stationary_grab(object(), self.cfg, execute=True)
        self.assertFalse(result["success"])
        self.assertEqual(result["state"], "reacquire")
        self.arm.stop.assert_awaited_once()
        self.gripper.grab.assert_not_awaited()

    async def test_target_shift_aborts_before_descent(self):
        moved = BallPose(BALL.x+30, BALL.y, BALL.z, "world", "ball", 100)
        self.observe.side_effect = [(BALL, DETECTION), (moved, DETECTION)]
        result = await run_stationary_grab(object(), self.cfg, execute=True)
        self.assertFalse(result["success"])
        self.assertIn("Ball moved", result["reason"])
        self.gripper.open.assert_not_awaited()

    async def test_failed_grasp_never_lifts(self):
        self.gripper.grab.side_effect = None
        self.gripper.grab.return_value = False
        result = await run_stationary_grab(object(), self.cfg, execute=True)
        self.assertFalse(result["success"])
        self.assertEqual(result["state"], "grab")
        self.assertEqual(self.motion.move.await_count, 4)
        self.arm.stop.assert_awaited_once()

    async def test_motion_failure_and_timeout_stop_hardware(self):
        self.motion.move.side_effect = TimeoutError("motion timeout")
        result = await run_stationary_grab(object(), self.cfg, execute=True)
        self.assertFalse(result["success"])
        self.arm.stop.assert_awaited_once()
        self.gripper.stop.assert_awaited_once()
        self.gripper.grab.assert_not_awaited()

    async def test_cancelled_motion_stops_and_propagates_cancellation(self):
        self.motion.move.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await run_stationary_grab(object(), self.cfg, execute=True)
        self.arm.stop.assert_awaited_once()
        self.gripper.stop.assert_awaited_once()


class ObservationTests(unittest.IsolatedAsyncioTestCase):
    async def observe(self, positions, joints=None, count=1):
        cfg = config()
        arm = SimpleNamespace(is_moving=AsyncMock(return_value=False), get_joint_positions=AsyncMock(
            side_effect=[SimpleNamespace(values=j) for j in (joints or [[0]*6, [0]*6])]))
        objects = [PointCloudObject(point_cloud=b"cloud", geometries=GeometriesInFrame(
            reference_frame="cam", geometries=[Geometry(label="ball", center=Pose(x=10, y=20, z=700))]))]*count
        service = SimpleNamespace(get_object_point_clouds=AsyncMock(return_value=objects))
        machine = SimpleNamespace(transform_pose=AsyncMock(side_effect=[PoseInFrame(
            reference_frame="world", pose=Pose(x=x, y=-818, z=18)) for x in positions]))
        with patch("motion.viam_stationary_grab.VisionClient.from_robot", return_value=service), \
             patch("motion.viam_stationary_grab.detect_ball", AsyncMock(return_value=DETECTION)), \
             patch("motion.viam_stationary_grab.asyncio.sleep", AsyncMock()):
            return await _observe(machine, arm, cfg)

    async def test_stationary_observation_uses_world_transform(self):
        ball, _ = await self.observe([198, 198, 198])
        self.assertEqual((ball.x, ball.reference_frame), (198, "world"))

    async def test_moving_ball_and_moving_wrist_rejected(self):
        with self.assertRaisesRegex(ValueError, "unstable"):
            await self.observe([198, 210, 220])
        with self.assertRaisesRegex(ValueError, "Arm moved"):
            await self.observe([198, 198, 198], joints=[[0]*6, [1]*6])

    async def test_multiple_balls_not_arbitrarily_selected(self):
        with self.assertRaisesRegex(ValueError, "found 2"):
            await self.observe([198]*3, count=2)


if __name__ == "__main__":
    unittest.main()
