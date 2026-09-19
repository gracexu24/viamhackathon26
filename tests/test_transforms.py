import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vision.transforms import RigidTransform, load_transform


ROOT = Path(__file__).resolve().parents[1]


class RigidTransformTests(unittest.TestCase):
    def test_cam2_calibration_maps_origin_and_axes(self):
        transform = load_transform(ROOT / "cam2_transform.json")
        self.assertEqual(transform.source_frame, "cam2")
        self.assertEqual(transform.destination_frame, "world")
        np.testing.assert_allclose(
            transform.point([0, 0, 0]), [-1128.165, -1179.987, 463.031], atol=1e-6)
        np.testing.assert_allclose(
            transform.vector([1, 0, 0]), [0.995699, 0.030140, -0.087609], atol=1e-6)

    def test_viam_quaternion_matches_matrix_rotation(self):
        payload = json.loads((ROOT / "cam2_transform.json").read_text(encoding="utf-8"))
        value = payload["viam_frame"]["orientation"]["value"]
        w, x, y, z = (value[key] for key in ("w", "x", "y", "z"))
        rotation = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ])
        np.testing.assert_allclose(
            rotation, load_transform(ROOT / "cam2_transform.json").rotation, atol=1e-6)

    def test_inverse_round_trip(self):
        transform = load_transform(ROOT / "cam2_transform.json")
        point = np.array([120.0, -35.0, 840.0])
        np.testing.assert_allclose(
            transform.inverse().point(transform.point(point)), point, atol=1e-3)

    def test_translation_is_not_applied_to_vectors(self):
        transform = load_transform(ROOT / "cam2_transform.json")
        np.testing.assert_allclose(
            np.asarray(transform.point([0, 0, 0])) + transform.vector([0, 0, 10]),
            transform.point([0, 0, 10]), atol=1e-6)

    def test_invalid_matrix_and_units_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "bottom row"):
            RigidTransform("camera", "world", np.ones((4, 4)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transform.json"
            path.write_text(json.dumps({
                "source_frame": "camera",
                "destination_frame": "world",
                "units": "meters",
                "matrix": np.eye(4).tolist(),
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "millimeters"):
                load_transform(path)


if __name__ == "__main__":
    unittest.main()
