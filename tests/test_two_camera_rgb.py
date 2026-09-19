import inspect
import unittest

import cv2
import numpy as np

import motion.two_camera_rgb_local as runtime
from motion.trajectory_fit import (
    FrontSample, PixelSample, fit_front_lateral, lateral_decision,
)
from vision.yellow_ball import (
    BallReleaseDetector, PixelMeasurement, TemporalPixelTracker,
    YellowCandidate, detect_yellow_candidates,
)


def candidate(u, v, radius=10):
    return YellowCandidate(u, v, radius, np.pi*radius**2, 0.9)


class YellowDetectionTests(unittest.TestCase):
    def test_yellow_ball_detected_and_multiple_candidates_returned(self):
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.circle(image, (75, 90), 14, (0, 255, 255), -1)
        cv2.circle(image, (220, 100), 10, (0, 255, 255), -1)
        found = detect_yellow_candidates(image)
        self.assertEqual(len(found), 2)
        self.assertAlmostEqual(found[0].u, 75, delta=1)


class TemporalTrackerTests(unittest.TestCase):
    def test_one_miss_does_not_clear_track_or_add_measurement(self):
        tracker = TemporalPixelTracker(max_gap_s=.1, association_gate_px=30,
                                       velocity_smoothing=1)
        first = tracker.update(1.00, [candidate(100, 80)])
        self.assertIsNotNone(first)
        track_id = tracker.track_id
        self.assertIsNone(tracker.update(1.04, []))
        self.assertTrue(tracker.is_active(1.04))
        second = tracker.update(1.08, [candidate(108, 80)])
        self.assertIsNotNone(second)
        self.assertEqual(tracker.track_id, track_id)
        self.assertAlmostEqual(tracker.velocity_px_s[0], 100)

    def test_track_expires_only_after_max_gap(self):
        tracker = TemporalPixelTracker(max_gap_s=.1)
        tracker.update(2.0, [candidate(50, 50)])
        self.assertTrue(tracker.is_active(2.1))
        tracker.update(2.101, [])
        self.assertFalse(tracker.is_active(2.101))
        self.assertIsNone(tracker.last_measurement)


class BallReleaseDetectorTests(unittest.TestCase):
    @staticmethod
    def measurement(timestamp, u, v=100):
        return PixelMeasurement(timestamp, u, v, 10)

    def test_stationary_ball_becomes_held_without_releasing(self):
        detector = BallReleaseDetector(40, 150, 2, .15)
        events = [detector.update(self.measurement(i*.05, 100)) for i in range(8)]
        self.assertEqual(detector.state, detector.HELD)
        self.assertFalse(detector.released)
        self.assertTrue(all(event is None for event in events))

    def test_small_hand_jitter_does_not_release(self):
        detector = BallReleaseDetector(40, 150, 2, .15)
        points = [100, 100.5, 99.5, 100.5, 100, 101, 100.5, 100]
        events = [detector.update(self.measurement(i*.05, u))
                  for i, u in enumerate(points)]
        self.assertEqual(detector.state, detector.HELD)
        self.assertTrue(all(event is None for event in events))

    def test_release_fires_once_and_keeps_first_fast_measurements(self):
        detector = BallReleaseDetector(40, 150, 2, .15)
        for i in range(5):
            self.assertIsNone(detector.update(self.measurement(i*.05, 100)))
        first_fast = self.measurement(.25, 110)
        second_fast = self.measurement(.30, 122)
        self.assertIsNone(detector.update(first_fast))
        event = detector.update(second_fast)
        self.assertIsNotNone(event)
        self.assertEqual(event.release_id, 1)
        self.assertEqual(event.timestamp, first_fast.timestamp)
        self.assertEqual(event.flight_measurements, (first_fast, second_fast))
        for measurement in (self.measurement(.35, 136), self.measurement(.40, 152)):
            self.assertIsNone(detector.update(measurement))

    def test_reset_allows_next_track_and_increments_release_id(self):
        detector = BallReleaseDetector(40, 150, 2, .10)

        def throw(offset):
            for i in range(4):
                detector.update(self.measurement(offset+i*.05, 100))
            detector.update(self.measurement(offset+.20, 110))
            return detector.update(self.measurement(offset+.25, 122))

        self.assertEqual(throw(0).release_id, 1)
        detector.reset()
        self.assertEqual(detector.state, detector.UNKNOWN)
        self.assertEqual(throw(1).release_id, 2)


class PredictionTests(unittest.TestCase):
    def test_side_fit_predicts_known_future_crossing(self):
        samples = [PixelSample(i*.02, 100+400*i*.02, 200-50*i*.02+200*(i*.02)**2)
                   for i in range(7)]
        prediction = runtime.predict_side_catch(samples, 180, .5, 8)
        self.assertAlmostEqual(prediction.catch_timestamp, .2, places=5)

    def test_past_side_crossing_rejected(self):
        samples = [PixelSample(i*.02, 200+200*i*.02, 100+i) for i in range(7)]
        with self.assertRaisesRegex(ValueError, "behind"):
            runtime.predict_side_catch(samples, 150, .5, 8)

    def test_side_trajectory_moving_away_is_rejected(self):
        samples = [PixelSample(i*.02, 200-200*i*.02, 100+i) for i in range(7)]
        with self.assertRaisesRegex(ValueError, "behind"):
            runtime.predict_side_catch(samples, 250, .5, 8)

    def test_side_trajectory_with_excessive_fit_error_is_rejected(self):
        samples = [PixelSample(i*.02, 100+20*(i % 2), 200+i) for i in range(7)]
        with self.assertRaisesRegex(ValueError, "Unstable image trajectory"):
            runtime.predict_side_catch(samples, 180, .5, 1)

    def test_front_linear_fit_recovers_horizontal_velocity(self):
        samples = [FrontSample(i*.03, 300+120*i*.03) for i in range(5)]
        fit = fit_front_lateral(samples)
        self.assertAlmostEqual(fit.du_px_s, 120, places=5)

    def test_front_linear_fit_predicts_future_u(self):
        samples = [FrontSample(i*.03, 300+120*i*.03) for i in range(5)]
        fit = fit_front_lateral(samples)
        self.assertAlmostEqual(fit.predict_u(.3), 336, places=5)

    def test_front_linear_fit_predicts_u_and_v_at_same_timestamp(self):
        samples = [FrontSample(i*.03, 300+120*i*.03, 220-80*i*.03)
                   for i in range(5)]
        fit = fit_front_lateral(samples)
        self.assertAlmostEqual(fit.predict_u(.3), 336, places=5)
        self.assertAlmostEqual(fit.predict_v(.3), 196, places=5)

    def test_left_of_basket_returns_left(self):
        self.assertEqual(lateral_decision(-30, 10), "LEFT")

    def test_inside_deadband_returns_center(self):
        self.assertEqual(lateral_decision(8, 10), "CENTER")

    def test_right_of_basket_returns_right(self):
        self.assertEqual(lateral_decision(30, 10), "RIGHT")

    def test_lateral_inversion_reverses_left_and_right(self):
        self.assertEqual(lateral_decision(-30, 10, invert=True), "RIGHT")
        self.assertEqual(lateral_decision(30, 10, invert=True), "LEFT")

    def test_combined_prediction_uses_exact_side_catch_timestamp(self):
        side_samples = [PixelSample(i*.02, 100+400*i*.02, 200) for i in range(7)]
        side = runtime.predict_side_catch(side_samples, 180, .5, 8)
        front_samples = [FrontSample(i*.03, 300+100*i*.03) for i in range(5)]
        front = fit_front_lateral(front_samples)
        prediction = runtime.combine_prediction(side, len(front_samples), front, 300, 10)
        self.assertAlmostEqual(prediction.predicted_ball_u_at_catch, 320, places=5)
        self.assertAlmostEqual(prediction.lateral_error_px, 20, places=5)
        self.assertEqual(prediction.decision, "RIGHT")

    def test_combined_prediction_contains_vertical_error(self):
        side_samples = [PixelSample(i*.02, 100+400*i*.02, 200) for i in range(7)]
        side = runtime.predict_side_catch(side_samples, 180, .5, 8)
        samples = [FrontSample(i*.03, 300+100*i*.03, 200+50*i*.03)
                   for i in range(5)]
        fit = fit_front_lateral(samples)
        prediction = runtime.combine_prediction(
            side, len(samples), fit, 300, 10, basket_v_px=190,
            vertical_deadband_px=10)
        self.assertAlmostEqual(prediction.predicted_front_v_at_catch, 210, places=5)
        self.assertAlmostEqual(prediction.vertical_error_px, 20, places=5)

    def test_runtime_contains_no_robot_motion_api(self):
        source = inspect.getsource(runtime)
        for forbidden in ("viam.components.arm", "MotionClient", ".move_to_", ".move(",
                          "Gripper", "point_cloud", "depth_source"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
