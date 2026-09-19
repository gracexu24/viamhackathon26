import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from viam.proto.common import Pose, PoseInFrame

from motion.move_to_trajectory import (
    closest_point_on_trajectory,
    closest_point_on_world_flight,
    load_trajectory,
    move_to_trajectory,
    validate_trajectory,
)
from motion.trajectory_fit import WorldFlight


class GeometryTests(unittest.TestCase):
    def test_closest_point_can_be_inside_segment(self):
        result = closest_point_on_trajectory((5, 3, 0), [(0, 0, 0), (10, 0, 0), (10, 10, 0)])
        self.assertEqual(result.segment_index, 0)
        self.assertAlmostEqual(result.segment_fraction, 0.5)
        self.assertEqual(result.xyz, (5.0, 0.0, 0.0))
        self.assertAlmostEqual(result.distance_mm, 3.0)

    def test_projection_is_clamped_to_endpoint(self):
        result = closest_point_on_trajectory((-2, 0, 0), [(0, 0, 0), (10, 0, 0)])
        self.assertEqual(result.xyz, (0.0, 0.0, 0.0))
        self.assertEqual(result.segment_fraction, 0.0)
        self.assertAlmostEqual(result.distance_mm, 2.0)

    def test_duplicate_points_are_safe(self):
        result = closest_point_on_trajectory((1, 0, 0), [(0, 0, 0), (0, 0, 0), (5, 0, 0)])
        self.assertEqual(result.segment_index, 1)
        self.assertEqual(result.xyz, (1.0, 0.0, 0.0))

    def test_invalid_trajectory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            validate_trajectory([(0, 0, 0)])
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_trajectory([(0, 0, 0), (math.nan, 1, 2)])

    def test_json_loader_accepts_object_and_checks_frame(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "trajectory.json"
            path.write_text(json.dumps({"reference_frame": "world", "points": [[0, 0, 0], [1, 2, 3]]}))
            self.assertEqual(load_trajectory(path), ((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)))
            path.write_text(json.dumps({"reference_frame": "camera", "points": [[0, 0, 0], [1, 2, 3]]}))
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_trajectory(path)

    def test_closest_point_on_world_flight(self):
        flight = WorldFlight(
            timestamp=10.0,
            position_mm=(0, 0, 0),
            velocity_mm_s=(100, 0, 0),
            acceleration_mm_s2=(0, 0, 0),
            rms_error_mm=1,
        )
        result = closest_point_on_world_flight((25, 10, 0), flight, horizon_s=1)
        self.assertAlmostEqual(result.trajectory_time_s, 0.25)
        self.assertEqual(result.xyz, (25.0, 0.0, 0.0))
        self.assertAlmostEqual(result.distance_mm, 10)

    def test_json_loader_accepts_world_flight(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "flight.json"
            path.write_text(json.dumps({
                "reference_frame": "world",
                "timestamp": 10,
                "position_mm": [1, 2, 3],
                "velocity_mm_s": [4, 5, 6],
                "acceleration_mm_s2": [0, 0, -9810],
                "rms_error_mm": 2,
            }))
            flight = load_trajectory(path)
            self.assertIsInstance(flight, WorldFlight)
            self.assertEqual(flight.velocity_mm_s, (4.0, 5.0, 6.0))


class MotionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.arm = SimpleNamespace(stop=AsyncMock())
        start = PoseInFrame(
            reference_frame="world",
            pose=Pose(x=5, y=4, z=3, o_x=0, o_y=0, o_z=-1, theta=30),
        )
        final = PoseInFrame(
            reference_frame="world",
            pose=Pose(x=5, y=0, z=0, o_x=0, o_y=0, o_z=-1, theta=30),
        )
        self.motion = SimpleNamespace(
            get_pose=AsyncMock(side_effect=[start, final]),
            move=AsyncMock(return_value=True),
        )
        self.patches = (
            patch("motion.move_to_trajectory.Arm.from_robot", return_value=self.arm),
            patch("motion.move_to_trajectory.MotionClient.from_robot", return_value=self.motion),
        )
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    async def test_preview_reads_pose_but_does_not_move(self):
        result = await move_to_trajectory(object(), [(0, 0, 0), (10, 0, 0)])
        self.assertFalse(result["executed"])
        self.assertEqual(result["closest"]["xyz"], (5.0, 0.0, 0.0))
        self.motion.move.assert_not_awaited()
        self.arm.stop.assert_not_awaited()

    async def test_execute_preserves_orientation_and_verifies_arrival(self):
        result = await move_to_trajectory(
            object(), [(0, 0, 0), (10, 0, 0)], execute=True,
            workspace_mm=[[-10, -10, -10], [20, 20, 20]], arrival_tolerance_mm=1
        )
        self.assertTrue(result["executed"])
        destination = self.motion.move.await_args.kwargs["destination"]
        self.assertEqual((destination.pose.x, destination.pose.y, destination.pose.z), (5, 0, 0))
        self.assertEqual((destination.pose.o_x, destination.pose.o_y,
                          destination.pose.o_z, destination.pose.theta), (0, 0, -1, 30))
        self.arm.stop.assert_not_awaited()

    async def test_motion_failure_requests_arm_stop(self):
        self.motion.move.return_value = False
        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            await move_to_trajectory(
                object(), [(0, 0, 0), (10, 0, 0)], execute=True,
                workspace_mm=[[-10, -10, -10], [20, 20, 20]],
            )
        self.arm.stop.assert_awaited_once()

    async def test_workspace_rejects_target_before_motion(self):
        with self.assertRaisesRegex(ValueError, "outside workspace"):
            await move_to_trajectory(
                object(),
                [(0, 0, 0), (10, 0, 0)],
                workspace_mm=[[20, 20, 20], [30, 30, 30]],
                execute=True,
            )
        self.motion.move.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
