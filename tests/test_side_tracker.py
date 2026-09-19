import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from vision.side_tracker import capture_time, decode_color, detect_red_ball


class SideTrackerTests(unittest.TestCase):
    def test_one_red_ball_and_capture_timestamp(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(frame, (124, 82), 15, (0, 0, 255), -1)
        timestamp = capture_time(SimpleNamespace(captured_at=SimpleNamespace(seconds=42, nanos=250000000)))
        obs = detect_red_ball(frame, timestamp)
        self.assertAlmostEqual(obs.timestamp, 42.25)
        self.assertAlmostEqual(obs.u, 124, delta=1)
        self.assertAlmostEqual(obs.v, 82, delta=1)
        self.assertAlmostEqual(obs.radius_px, 15, delta=2)

    def test_ambiguous_red_balls_rejected(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(frame, (80, 80), 15, (0, 0, 255), -1)
        cv2.circle(frame, (220, 80), 15, (0, 0, 255), -1)
        self.assertIsNone(detect_red_ball(frame, 1.0))

    def test_decode_color(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode('.jpg', frame)
        self.assertTrue(ok)
        decoded = decode_color(SimpleNamespace(data=encoded.tobytes()))
        self.assertEqual(decoded.shape, frame.shape)


if __name__ == '__main__':
    unittest.main()
