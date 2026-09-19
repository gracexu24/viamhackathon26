"""Shared helpers for read-only local Viam camera processes."""

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return result


def credentials(config_path):
    """Read API credentials into memory without logging or copying them."""
    if config_path:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        handlers = config.get("auth", {}).get("handlers", [])
        for handler in handlers:
            if handler.get("type") == "api-key":
                for key_id, key in handler.get("config", {}).items():
                    if key_id != "keys" and isinstance(key, str):
                        return key_id, key
        raise ValueError("No API-key credentials found in machine config")
    return os.environ["VIAM_API_KEY_ID"], os.environ["VIAM_API_KEY"]


def decode_color(image):
    """Decode one Viam JPEG/PNG image into an OpenCV BGR array."""
    bgr = cv2.imdecode(np.frombuffer(image.data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Camera color image is not a decodable JPEG/PNG")
    return bgr
