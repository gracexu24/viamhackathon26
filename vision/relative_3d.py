"""Pure 3D perception logic for ball-to-basket alignment; never moves hardware.

Both points must come from the same aligned color/depth frame. Camera optical
axes are X right, Y down, Z away from the camera. An optional calibrated 3x3
rotation converts the relative vector to world axes; translation cancels because
the output is a difference between two points in the same camera frame.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Point3D:
    x_mm: float
    y_mm: float
    z_mm: float

    def array(self):
        return np.array([self.x_mm, self.y_mm, self.z_mm], dtype=float)


@dataclass(frozen=True)
class Guidance3D:
    frame: str
    delta_mm: tuple[float, float, float]
    distance_mm: float
    directions: tuple[str, ...]
    aligned: bool


@dataclass(frozen=True)
class Relative3DResult:
    ball_camera_mm: Point3D
    arm_reference_camera_mm: Point3D
    guidance: Guidance3D


def validate_intrinsics(intrinsics):
    values = [float(intrinsics[name]) for name in ("fx", "fy", "cx", "cy")]
    if not np.isfinite(values).all() or values[0] <= 0 or values[1] <= 0:
        raise ValueError("Camera intrinsics must be finite with positive focal lengths")
    return values


def deproject_pixel(u, v, depth_mm, intrinsics):
    """Deproject a color pixel using depth aligned to that color image."""
    fx, fy, cx, cy = validate_intrinsics(intrinsics)
    values = np.asarray([u, v, depth_mm], dtype=float)
    if not np.isfinite(values).all() or depth_mm <= 0:
        raise ValueError("Pixel and depth must be finite; depth must be positive")
    return Point3D((float(u) - cx) * float(depth_mm) / fx,
                   (float(v) - cy) * float(depth_mm) / fy,
                   float(depth_mm))


def robust_roi_depth(depth_image_mm, roi, *, min_depth_mm=150, max_depth_mm=5000,
                     min_valid_pixels=9):
    """Median depth in [left, top, right, bottom], rejecting holes and outliers."""
    depth = np.asarray(depth_image_mm, dtype=float)
    if depth.ndim != 2:
        raise ValueError("Depth image must be a 2D millimeter array")
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        raise ValueError("ROI must be [left, top, right, bottom]")
    left, top, right, bottom = (int(value) for value in roi)
    if not 0 <= left < right <= depth.shape[1] or not 0 <= top < bottom <= depth.shape[0]:
        raise ValueError("ROI is outside the depth image")
    values = depth[top:bottom, left:right].reshape(-1)
    values = values[np.isfinite(values) & (values >= min_depth_mm) & (values <= max_depth_mm)]
    if len(values) < min_valid_pixels:
        raise ValueError("Arm/basket ROI has too few valid depth pixels")
    median = float(np.median(values))
    deviations = np.abs(values - median)
    mad = float(np.median(deviations))
    tolerance = max(15.0, 3.0 * 1.4826 * mad)
    inliers = values[deviations <= tolerance]
    if len(inliers) < min_valid_pixels:
        raise ValueError("Arm/basket ROI depth is inconsistent")
    return float(np.median(inliers))


def point_from_depth_roi(depth_image_mm, roi, intrinsics, **depth_options):
    """Return the 3D surface point at the center of a tracked arm/basket ROI."""
    depth = robust_roi_depth(depth_image_mm, roi, **depth_options)
    left, top, right, bottom = (int(value) for value in roi)
    u = (left + right - 1) / 2
    v = (top + bottom - 1) / 2
    return deproject_pixel(u, v, depth, intrinsics)


def ball_center_point(observation):
    """Convert ball_tracking.Observation.xyz into a validated Point3D."""
    xyz = np.asarray(observation.xyz, dtype=float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all() or xyz[2] <= 0:
        raise ValueError("Ball observation must contain a valid camera-frame XYZ center")
    return Point3D(*map(float, xyz))


def relative_guidance(ball, arm_reference, *, deadband_mm=(25, 25, 40),
                      camera_to_world_rotation=None):
    """Describe how the arm reference would need to translate to meet the ball.

    This is a recommendation only. It performs no reachability, collision, timing,
    or motion planning and must not be sent directly to an arm controller.
    """
    ball_xyz = ball.array()
    arm_xyz = arm_reference.array()
    if not np.isfinite(ball_xyz).all() or not np.isfinite(arm_xyz).all():
        raise ValueError("Ball and arm points must be finite")
    deadband = np.asarray(deadband_mm, dtype=float)
    if deadband.shape != (3,) or not np.isfinite(deadband).all() or np.any(deadband < 0):
        raise ValueError("Deadband must contain three finite nonnegative millimeter values")
    delta = ball_xyz - arm_xyz
    frame = "camera_optical"
    if camera_to_world_rotation is not None:
        rotation = np.asarray(camera_to_world_rotation, dtype=float)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all() \
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) \
                or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-3):
            raise ValueError("camera_to_world_rotation must be a proper 3x3 rotation")
        delta = rotation @ delta
        frame = "world"
    labels = (("right", "left"), ("down", "up"), ("away_from_camera", "toward_camera")) \
        if frame == "camera_optical" else (("world_+x", "world_-x"),
                                            ("world_+y", "world_-y"),
                                            ("up", "down"))
    directions = []
    for value, tolerance, (positive, negative) in zip(delta, deadband, labels):
        if value > tolerance:
            directions.append(positive)
        elif value < -tolerance:
            directions.append(negative)
    return Guidance3D(frame, tuple(map(float, delta)), float(np.linalg.norm(delta)),
                      tuple(directions), not directions)


def analyze_aligned_frame(color_bgr, depth_image_mm, intrinsics, ball_radius_mm,
                          timestamp, arm_reference_roi, *, deadband_mm=(25, 25, 40),
                          camera_to_world_rotation=None):
    """Ball center + arm reference surface -> read-only relative 3D guidance."""
    from ball_tracking import detect_red_ball

    observation = detect_red_ball(color_bgr, np.asarray(depth_image_mm, dtype=float),
                                  intrinsics, ball_radius_mm, timestamp)
    if observation is None:
        raise ValueError("No unambiguous red ball with valid aligned depth")
    ball = ball_center_point(observation)
    arm_reference = point_from_depth_roi(depth_image_mm, arm_reference_roi, intrinsics)
    guidance = relative_guidance(ball, arm_reference, deadband_mm=deadband_mm,
                                 camera_to_world_rotation=camera_to_world_rotation)
    return Relative3DResult(ball, arm_reference, guidance)
