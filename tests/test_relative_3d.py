import unittest
from types import SimpleNamespace

import numpy as np

from vision.relative_3d import (
    Point3D, analyze_aligned_frame, ball_center_point, deproject_pixel, point_from_depth_roi,
    relative_guidance, robust_roi_depth,
)


INTRINSICS = {"fx": 600, "fy": 600, "cx": 424, "cy": 240}


class Relative3DTests(unittest.TestCase):
    def test_deproject_and_robust_arm_roi(self):
        point = deproject_pixel(484, 270, 1000, INTRINSICS)
        self.assertEqual(point, Point3D(100, 50, 1000))
        depth = np.full((40, 60), 900.0)
        depth[15, 25] = 0
        depth[16, 26] = 4000
        self.assertEqual(robust_roi_depth(depth, [20, 10, 40, 30]), 900)
        roi_point = point_from_depth_roi(depth, [20, 10, 40, 30],
                                         {"fx": 100, "fy": 100, "cx": 30, "cy": 20})
        self.assertAlmostEqual(roi_point.z_mm, 900)

    def test_camera_direction_and_deadband(self):
        result = relative_guidance(Point3D(50, -30, 1100), Point3D(0, 0, 1000),
                                   deadband_mm=(20, 20, 20))
        self.assertEqual(result.frame, "camera_optical")
        self.assertEqual(result.directions, ("right", "up", "away_from_camera"))
        self.assertFalse(result.aligned)
        aligned = relative_guidance(Point3D(5, 5, 1010), Point3D(0, 0, 1000))
        self.assertTrue(aligned.aligned)

    def test_world_rotation_changes_axis_labels(self):
        # Camera X -> world X, camera Y down -> world -Z, camera Z -> world Y.
        rotation = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        result = relative_guidance(Point3D(40, 50, 1100), Point3D(0, 0, 1000),
                                   deadband_mm=(10, 10, 10),
                                   camera_to_world_rotation=rotation)
        self.assertEqual(result.delta_mm, (40.0, 100.0, -50.0))
        self.assertEqual(result.directions, ("world_+x", "world_+y", "down"))

    def test_ball_observation_conversion_and_bad_rotation(self):
        point = ball_center_point(SimpleNamespace(xyz=np.array([1, 2, 300])))
        self.assertEqual(point, Point3D(1, 2, 300))
        with self.assertRaisesRegex(ValueError, "proper"):
            relative_guidance(point, point, camera_to_world_rotation=np.eye(3) * 2)

    def test_full_aligned_frame_logic(self):
        import cv2
        color = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.circle(color, (80, 60), 9, (0, 0, 255), -1)
        depth = np.full((120, 160), 900.0)
        intrinsics = {"width": 160, "height": 120,
                      "fx": 250, "fy": 250, "cx": 80, "cy": 60}
        result = analyze_aligned_frame(color, depth, intrinsics, 20, 1.0,
                                       [100, 50, 120, 70], deadband_mm=(5, 5, 5))
        self.assertGreater(result.ball_camera_mm.z_mm, 900)  # sphere-center correction
        self.assertEqual(result.arm_reference_camera_mm.z_mm, 900)
        self.assertIn("left", result.guidance.directions)


if __name__ == "__main__":
    unittest.main()
