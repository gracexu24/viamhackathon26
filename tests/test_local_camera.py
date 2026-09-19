import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from vision.local_camera import credentials, decode_color, positive


class LocalCameraTests(unittest.TestCase):
    def test_positive_requires_finite_positive_value(self):
        self.assertEqual(positive("2.5"), 2.5)
        for value in ("0", "-1", "nan", "inf"):
            with self.assertRaises(argparse.ArgumentTypeError):
                positive(value)

    def test_credentials_from_config_or_environment(self):
        payload = {"auth": {"handlers": [{
            "type": "api-key",
            "config": {"key-id": "secret", "keys": ["unused"]},
        }]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "machine.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(credentials(path), ("key-id", "secret"))
        with patch.dict(os.environ, {"VIAM_API_KEY_ID": "env-id", "VIAM_API_KEY": "env-key"}):
            self.assertEqual(credentials(None), ("env-id", "env-key"))

    def test_decode_color(self):
        expected = np.zeros((8, 10, 3), dtype=np.uint8)
        expected[:, :, 1] = 200
        ok, encoded = cv2.imencode(".png", expected)
        self.assertTrue(ok)
        actual = decode_color(SimpleNamespace(data=encoded.tobytes()))
        np.testing.assert_array_equal(actual, expected)
        with self.assertRaisesRegex(ValueError, "decodable"):
            decode_color(SimpleNamespace(data=b"not an image"))


if __name__ == "__main__":
    unittest.main()
