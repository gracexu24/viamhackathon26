"""Short-horizon flight fits from capture-timestamped observations.

Pixel fits predict only image coordinates. A physical gravity model requires
calibrated positions in a Z-up world frame; never treat pixels as millimeters.
"""

from dataclasses import dataclass
import math

import numpy as np


GRAVITY_MM_S2 = (0.0, 0.0, -9810.0)


@dataclass(frozen=True)
class PixelSample:
    timestamp: float
    u: float
    v: float


@dataclass(frozen=True)
class FrontSample:
    timestamp: float
    u: float


@dataclass(frozen=True)
class FrontLateralFit:
    timestamp: float
    u_px: float
    du_px_s: float
    rms_error_px: float

    def predict_u(self, timestamp):
        return self.u_px + self.du_px_s * (float(timestamp)-self.timestamp)


def fit_front_lateral(samples, *, min_samples=4, min_span_s=0.04,
                      max_error_px=8.0):
    """Fit front-camera horizontal motion u(t)=u0+vu*t from real samples."""
    dt = _times(samples, int(min_samples), float(min_span_s))
    values = np.asarray([sample.u for sample in samples], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Front-camera samples must be finite")
    design = np.column_stack((np.ones(len(dt)), dt))
    u0, velocity = np.linalg.lstsq(design, values, rcond=None)[0]
    error = float(np.sqrt(np.mean((design @ [u0, velocity]-values)**2)))
    if not math.isfinite(error) or error > max_error_px:
        raise ValueError(f"Unstable front trajectory: RMS error {error:.1f} px")
    return FrontLateralFit(float(samples[-1].timestamp), float(u0),
                           float(velocity), error)


def lateral_decision(error_px, deadband_px, invert=False):
    """Classify signed image error; invert swaps labels, not error sign."""
    error_px, deadband_px = float(error_px), float(deadband_px)
    if not math.isfinite(error_px) or not math.isfinite(deadband_px) or deadband_px < 0:
        raise ValueError("Lateral error/deadband must be finite and deadband nonnegative")
    if abs(error_px) <= deadband_px:
        return "CENTER"
    decision = "LEFT" if error_px < 0 else "RIGHT"
    if invert:
        decision = "RIGHT" if decision == "LEFT" else "LEFT"
    return decision


@dataclass(frozen=True)
class PixelFlight:
    timestamp: float
    u_px: float
    v_px: float
    du_px_s: float
    dv_px_s: float
    d2v_px_s2: float
    rms_error_px: float

    def predict(self, timestamp):
        dt = float(timestamp) - self.timestamp
        return (self.u_px + self.du_px_s * dt,
                self.v_px + self.dv_px_s * dt + 0.5 * self.d2v_px_s2 * dt * dt)

    def crossing(self, u_px, max_horizon_s=0.5):
        """Future time and image height at a measured image catch line."""
        if abs(self.du_px_s) < 1:
            raise ValueError("Horizontal image motion is too small for a crossing prediction")
        dt = (float(u_px) - self.u_px) / self.du_px_s
        if not 0 < dt <= max_horizon_s:
            raise ValueError("Catch line is behind the ball or beyond prediction horizon")
        _, v = self.predict(self.timestamp + dt)
        return self.timestamp + dt, v


def _times(samples, min_samples, min_span_s):
    if len(samples) < min_samples:
        raise ValueError(f"Need at least {min_samples} observations")
    times = np.asarray([sample.timestamp for sample in samples], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Capture timestamps must be finite and strictly increasing")
    if times[-1] - times[0] < min_span_s:
        raise ValueError("Observation time span is too short")
    return times - times[-1]


def fit_pixel_flight(samples, *, max_error_px=8.0):
    """Fit u(t)=u0+vu*t and v(t)=v0+vv*t+0.5*av*t² in image space."""
    dt = _times(samples, 5, 0.06)
    uv = np.asarray([(sample.u, sample.v) for sample in samples], dtype=float)
    if not np.isfinite(uv).all():
        raise ValueError("Pixel observations must be finite")
    horizontal = np.column_stack((np.ones(len(dt)), dt))
    vertical = np.column_stack((np.ones(len(dt)), dt, 0.5 * dt**2))
    u0, vu = np.linalg.lstsq(horizontal, uv[:, 0], rcond=None)[0]
    v0, vv, av = np.linalg.lstsq(vertical, uv[:, 1], rcond=None)[0]
    predicted = np.column_stack((horizontal @ [u0, vu], vertical @ [v0, vv, av]))
    error = float(np.sqrt(np.mean(np.sum((predicted - uv)**2, axis=1))))
    if not math.isfinite(error) or error > max_error_px:
        raise ValueError(f"Unstable image trajectory: RMS error {error:.1f} px")
    return PixelFlight(float(samples[-1].timestamp), float(u0), float(v0),
                       float(vu), float(vv), float(av), error)


@dataclass(frozen=True)
class WorldSample:
    timestamp: float
    xyz_mm: tuple[float, float, float]


@dataclass(frozen=True)
class PlaneSample:
    timestamp: float
    x_mm: float
    z_mm: float


def calibrate_plane_homography(image_points_px, plane_points_mm):
    """Map an angled camera image onto one measured vertical throw plane."""
    image = np.asarray(image_points_px, dtype=float)
    plane = np.asarray(plane_points_mm, dtype=float)
    if image.ndim != 2 or image.shape[1:] != (2,) or image.shape != plane.shape \
            or len(image) < 4 or not np.isfinite(image).all() or not np.isfinite(plane).all():
        raise ValueError("Need at least four finite matching image and plane XY points")
    rows = []
    for (u, v), (x, z) in zip(image, plane):
        rows.extend(([-u, -v, -1, 0, 0, 0, u*x, v*x, x],
                     [0, 0, 0, -u, -v, -1, u*z, v*z, z]))
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=float))
    matrix = vh[-1].reshape(3, 3)
    if abs(matrix[2, 2]) < 1e-12 or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("Plane calibration points are degenerate")
    return matrix / matrix[2, 2]


def pixel_to_plane(u, v, homography):
    matrix = np.asarray(homography, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Plane homography must be a finite 3x3 matrix")
    mapped = matrix @ np.array([float(u), float(v), 1.0])
    if not np.isfinite(mapped).all() or abs(mapped[2]) < 1e-12:
        raise ValueError("Pixel maps to an invalid point on the throw plane")
    return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])


@dataclass(frozen=True)
class PlaneFlight:
    timestamp: float
    x_mm: float
    z_mm: float
    vx_mm_s: float
    vz_mm_s: float
    gravity_mm_s2: float
    rms_error_mm: float

    def predict(self, timestamp):
        dt = float(timestamp) - self.timestamp
        return (self.x_mm + self.vx_mm_s * dt,
                self.z_mm + self.vz_mm_s * dt + 0.5 * self.gravity_mm_s2 * dt**2)

    def crossing(self, x_mm, max_horizon_s=0.5):
        if abs(self.vx_mm_s) < 1:
            raise ValueError("Horizontal motion is too small for a crossing prediction")
        dt = (float(x_mm) - self.x_mm) / self.vx_mm_s
        if not 0 < dt <= max_horizon_s:
            raise ValueError("Physical catch line is behind the ball or beyond prediction horizon")
        _, z = self.predict(self.timestamp + dt)
        return self.timestamp + dt, z


def fit_plane_ballistic(samples, *, gravity_mm_s2=-9810.0, max_error_mm=30.0):
    """Fit horizontal motion and gravity-constrained height in a calibrated plane."""
    dt = _times(samples, 4, 0.06)
    xz = np.asarray([(sample.x_mm, sample.z_mm) for sample in samples], dtype=float)
    if not np.isfinite(xz).all() or not math.isfinite(gravity_mm_s2):
        raise ValueError("Plane observations and gravity must be finite")
    design = np.column_stack((np.ones(len(dt)), dt))
    x0, vx = np.linalg.lstsq(design, xz[:, 0], rcond=None)[0]
    corrected_z = xz[:, 1] - 0.5 * gravity_mm_s2 * dt**2
    z0, vz = np.linalg.lstsq(design, corrected_z, rcond=None)[0]
    predicted = np.column_stack((design @ [x0, vx], design @ [z0, vz] +
                                 0.5 * gravity_mm_s2 * dt**2))
    error = float(np.sqrt(np.mean(np.sum((predicted - xz)**2, axis=1))))
    if not math.isfinite(error) or error > max_error_mm:
        raise ValueError(f"Unstable plane trajectory: RMS error {error:.1f} mm")
    return PlaneFlight(float(samples[-1].timestamp), float(x0), float(z0), float(vx),
                       float(vz), float(gravity_mm_s2), error)


@dataclass(frozen=True)
class WorldFlight:
    timestamp: float
    position_mm: tuple[float, float, float]
    velocity_mm_s: tuple[float, float, float]
    acceleration_mm_s2: tuple[float, float, float]
    rms_error_mm: float

    def predict(self, timestamp):
        dt = float(timestamp) - self.timestamp
        return tuple(p + v * dt + 0.5 * a * dt**2 for p, v, a in
                     zip(self.position_mm, self.velocity_mm_s, self.acceleration_mm_s2))


def fit_world_ballistic(samples, *, gravity_mm_s2=GRAVITY_MM_S2, max_error_mm=30.0):
    """Fit p,v with known gravity from calibrated Z-up world XYZ samples."""
    dt = _times(samples, 4, 0.06)
    xyz = np.asarray([sample.xyz_mm for sample in samples], dtype=float)
    gravity = np.asarray(gravity_mm_s2, dtype=float)
    if xyz.shape != (len(samples), 3) or gravity.shape != (3,) or not np.isfinite(xyz).all() \
            or not np.isfinite(gravity).all():
        raise ValueError("Expected finite world XYZ and gravity vectors")
    corrected = xyz - 0.5 * dt[:, None]**2 * gravity
    design = np.column_stack((np.ones(len(dt)), dt))
    position, velocity = np.linalg.lstsq(design, corrected, rcond=None)[0]
    predicted = design @ np.vstack((position, velocity)) + 0.5 * dt[:, None]**2 * gravity
    error = float(np.sqrt(np.mean(np.sum((predicted - xyz)**2, axis=1))))
    if not math.isfinite(error) or error > max_error_mm:
        raise ValueError(f"Unstable world trajectory: RMS error {error:.1f} mm")
    return WorldFlight(float(samples[-1].timestamp), tuple(map(float, position)),
                       tuple(map(float, velocity)), tuple(map(float, gravity)), error)
