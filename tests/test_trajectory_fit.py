import unittest

from motion.trajectory_fit import (
    GRAVITY_MM_S2, PixelSample, PlaneSample, WorldSample,
    calibrate_plane_homography, fit_pixel_flight, fit_plane_ballistic,
    fit_world_ballistic, pixel_to_plane,
)


class PixelFlightTests(unittest.TestCase):
    def test_fits_parabola_and_predicts_catch_line(self):
        # Pixel v points down, so this synthetic image acceleration is positive.
        samples = []
        for i in range(8):
            t = i * 0.02
            samples.append(PixelSample(t, 100 + 300*t, 200 - 100*t + 400*t*t))
        fit = fit_pixel_flight(samples)
        self.assertAlmostEqual(fit.du_px_s, 300, places=5)
        self.assertAlmostEqual(fit.d2v_px_s2, 800, places=4)
        crossing_t, crossing_v = fit.crossing(160, max_horizon_s=0.5)
        self.assertAlmostEqual(crossing_t, 0.2, places=5)
        self.assertAlmostEqual(crossing_v, 196, places=4)
        self.assertLess(fit.rms_error_px, 1e-6)

    def test_rejects_bad_time_and_distant_crossing(self):
        duplicated = [PixelSample(0, 0, 0)] * 5
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            fit_pixel_flight(duplicated)
        fit = fit_pixel_flight([PixelSample(i*.02, i*10, i*i) for i in range(5)])
        with self.assertRaisesRegex(ValueError, "horizon"):
            fit.crossing(1000, max_horizon_s=.1)


class WorldFlightTests(unittest.TestCase):
    def test_known_gravity_fit(self):
        position = (100, -800, 600)
        velocity = (400, 120, 2000)
        samples = []
        for i in range(7):
            t = i * .02
            xyz = tuple(p + v*t + .5*a*t*t for p, v, a in
                        zip(position, velocity, GRAVITY_MM_S2))
            samples.append(WorldSample(t, xyz))
        fit = fit_world_ballistic(samples)
        predicted = fit.predict(.2)
        expected = tuple(p + v*.2 + .5*a*.2*.2 for p, v, a in
                         zip(position, velocity, GRAVITY_MM_S2))
        for actual, wanted in zip(predicted, expected):
            self.assertAlmostEqual(actual, wanted, places=5)
        self.assertLess(fit.rms_error_mm, 1e-6)


class PlaneFlightTests(unittest.TestCase):
    def test_angle_mapping_and_gravity_fit(self):
        image = [(100, 100), (500, 100), (100, 400), (500, 400)]
        plane = [(0, 1000), (2000, 1000), (0, 0), (2000, 0)]
        homography = calibrate_plane_homography(image, plane)
        x, z = pixel_to_plane(300, 250, homography)
        self.assertAlmostEqual(x, 1000, places=5)
        self.assertAlmostEqual(z, 500, places=5)
        samples = []
        for i in range(6):
            t = i * .02
            samples.append(PlaneSample(t, 100 + 500*t, 600 + 1800*t - 4905*t*t))
        fit = fit_plane_ballistic(samples)
        self.assertAlmostEqual(fit.vx_mm_s, 500, places=5)
        self.assertAlmostEqual(fit.vz_mm_s, 1800 - 9810*.1, places=5)
        crossing_t, height = fit.crossing(180, .5)
        self.assertAlmostEqual(crossing_t, .16, places=5)
        self.assertAlmostEqual(height, 600 + 1800*.16 - 4905*.16**2, places=4)


if __name__ == "__main__":
    unittest.main()
