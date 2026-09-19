"""Reusable HSV yellow-ball detection and lightweight pixel tracking."""

from dataclasses import dataclass
import math

import cv2
import numpy as np


DEFAULT_YELLOW_CONFIG = {
    "hue_min": 18,
    "hue_max": 40,
    "saturation_min": 90,
    "value_min": 80,
    "min_area_px": 60.0,
    "min_circularity": 0.55,
}


@dataclass(frozen=True)
class YellowCandidate:
    u: float
    v: float
    radius_px: float
    area_px: float
    circularity: float


@dataclass(frozen=True)
class PixelMeasurement:
    timestamp: float
    u: float
    v: float
    radius_px: float


def detect_yellow_candidates(bgr, config=None):
    """Return every plausible yellow circular blob, largest first."""
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected a BGR color image")
    settings = {**DEFAULT_YELLOW_CONFIG, **(config or {})}
    hue_min = int(settings["hue_min"])
    hue_max = int(settings["hue_max"])
    saturation_min = int(settings["saturation_min"])
    value_min = int(settings["value_min"])
    min_area = float(settings["min_area_px"])
    min_circularity = float(settings["min_circularity"])
    if not 0 <= hue_min <= hue_max <= 179:
        raise ValueError("Yellow hue bounds must satisfy 0 <= min <= max <= 179")
    if not 0 <= saturation_min <= 255 or not 0 <= value_min <= 255:
        raise ValueError("Yellow saturation/value floors must lie in [0,255]")
    if not math.isfinite(min_area) or min_area <= 0 or not 0 < min_circularity <= 1:
        raise ValueError("Yellow area and circularity thresholds are invalid")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (hue_min, saturation_min, value_min),
                       (hue_max, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = bgr.shape[:2]
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area < min_area or perimeter <= 0:
            continue
        circularity = 4 * math.pi * area / perimeter**2
        if circularity < min_circularity:
            continue
        (u, v), radius = cv2.minEnclosingCircle(contour)
        if radius <= 0 or u-radius <= 0 or v-radius <= 0 \
                or u+radius >= width-1 or v+radius >= height-1:
            continue
        candidates.append(YellowCandidate(float(u), float(v), float(radius),
                                          area, float(circularity)))
    return sorted(candidates, key=lambda item: item.area_px, reverse=True)


class TemporalPixelTracker:
    """Associate real detections using a constant-velocity pixel prediction.

    Missing frames never create synthetic measurements. The last real track is
    retained for max_gap_s and expires only after that capture-time gap.
    """

    def __init__(self, max_gap_s=0.1, association_gate_px=80.0,
                 velocity_smoothing=0.5):
        self.max_gap_s = float(max_gap_s)
        self.association_gate_px = float(association_gate_px)
        self.velocity_smoothing = float(velocity_smoothing)
        if not math.isfinite(self.max_gap_s) or self.max_gap_s <= 0:
            raise ValueError("max_gap_s must be positive and finite")
        if not math.isfinite(self.association_gate_px) or self.association_gate_px <= 0:
            raise ValueError("association_gate_px must be positive and finite")
        if not 0 <= self.velocity_smoothing <= 1:
            raise ValueError("velocity_smoothing must lie in [0,1]")
        self.last_measurement = None
        self.velocity_px_s = np.zeros(2, dtype=float)
        self.last_radius_px = None
        self.last_real_measurement_time = None
        self.track_id = 0

    def is_active(self, timestamp):
        return (self.last_real_measurement_time is not None
                and 0 <= float(timestamp)-self.last_real_measurement_time
                <= self.max_gap_s + 1e-9)

    def predict(self, timestamp):
        if self.last_measurement is None or not self.is_active(timestamp):
            return None
        dt = float(timestamp) - self.last_measurement.timestamp
        return np.array([self.last_measurement.u, self.last_measurement.v]) \
            + self.velocity_px_s * dt

    def _expire_if_needed(self, timestamp):
        if self.last_real_measurement_time is not None \
                and float(timestamp)-self.last_real_measurement_time > self.max_gap_s + 1e-9:
            self.last_measurement = None
            self.velocity_px_s = np.zeros(2, dtype=float)
            self.last_radius_px = None
            self.last_real_measurement_time = None

    def update(self, timestamp, candidates):
        """Return an accepted real measurement, or None for a miss."""
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("Tracker timestamp must be finite")
        self._expire_if_needed(timestamp)
        candidates = list(candidates)
        if not candidates:
            return None
        if self.last_measurement is None:
            selected = max(candidates, key=lambda item: item.area_px)
            self.track_id += 1
            velocity = np.zeros(2, dtype=float)
        else:
            predicted = self.predict(timestamp)
            def association_score(item):
                position_error = float(np.linalg.norm(np.array([item.u, item.v])-predicted))
                radius_penalty = (abs(math.log(item.radius_px/self.last_radius_px))*20
                                  if self.last_radius_px and item.radius_px > 0 else 0)
                return position_error + radius_penalty
            ranked = sorted(candidates, key=association_score)
            selected = ranked[0]
            if np.linalg.norm(np.array([selected.u, selected.v])-predicted) \
                    > self.association_gate_px:
                return None
            dt = timestamp - self.last_measurement.timestamp
            if dt <= 0:
                raise ValueError("Tracker measurement timestamps must increase")
            measured_velocity = (np.array([selected.u, selected.v]) - np.array([
                self.last_measurement.u, self.last_measurement.v])) / dt
            alpha = self.velocity_smoothing
            velocity = alpha*measured_velocity + (1-alpha)*self.velocity_px_s
        measurement = PixelMeasurement(timestamp, selected.u, selected.v,
                                       selected.radius_px)
        self.last_measurement = measurement
        self.velocity_px_s = velocity
        self.last_radius_px = selected.radius_px
        self.last_real_measurement_time = timestamp
        return measurement
