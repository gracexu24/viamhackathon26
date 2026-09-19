"""Validated rigid transforms for calibrated camera-frame 3D points.

Matrices use column vectors and the naming convention
``T_destination_from_source``. For example, ``T_world_from_cam2`` maps a point
measured in the cam2 optical frame into the world frame. Translation is in mm.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RigidTransform:
    source_frame: str
    destination_frame: str
    matrix: np.ndarray

    def __post_init__(self):
        if not self.source_frame or not self.destination_frame:
            raise ValueError("Transform source and destination frames are required")
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("Rigid transform must be a finite 4x4 matrix")
        if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9):
            raise ValueError("Rigid transform bottom row must be [0, 0, 0, 1]")
        rotation = matrix[:3, :3]
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)):
            raise ValueError("Rigid transform rotation must be orthonormal with determinant +1")
        object.__setattr__(self, "matrix", matrix)

    @property
    def rotation(self):
        return self.matrix[:3, :3].copy()

    @property
    def translation_mm(self):
        return self.matrix[:3, 3].copy()

    def point(self, xyz_mm):
        """Transform one absolute XYZ point, including translation."""
        xyz = _xyz(xyz_mm, "point")
        return tuple(map(float, self.matrix[:3, :3] @ xyz + self.matrix[:3, 3]))

    def vector(self, xyz):
        """Rotate a direction or relative displacement; translation is omitted."""
        value = _xyz(xyz, "vector")
        return tuple(map(float, self.matrix[:3, :3] @ value))

    def inverse(self):
        """Return the destination-to-source transform."""
        rotation = self.matrix[:3, :3]
        inverse = np.eye(4)
        inverse[:3, :3] = rotation.T
        inverse[:3, 3] = -rotation.T @ self.matrix[:3, 3]
        return RigidTransform(self.destination_frame, self.source_frame, inverse)


def _xyz(value, label):
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"Transform {label} must contain three finite values")
    return array


def load_transform(path):
    """Load a RigidTransform from a repository calibration JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("units") != "millimeters":
        raise ValueError("Transform units must be 'millimeters'")
    return RigidTransform(
        source_frame=str(payload.get("source_frame", "")),
        destination_frame=str(payload.get("destination_frame", "")),
        matrix=payload.get("matrix"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--point", type=float, nargs=3, metavar=("X", "Y", "Z"), required=True,
                        help="Source-frame XYZ in millimeters")
    args = parser.parse_args()
    transform = load_transform(args.config)
    result = transform.point(args.point)
    print(json.dumps({
        "source_frame": transform.source_frame,
        "destination_frame": transform.destination_frame,
        "source_point_mm": args.point,
        "destination_point_mm": result,
    }, indent=2))


if __name__ == "__main__":
    main()
