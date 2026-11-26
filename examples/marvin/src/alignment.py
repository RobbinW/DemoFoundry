"""Measure the three visible cube silhouettes in the recorded and proxy first frames."""

import cv2
import numpy as np


def cube_mask(hsv, name):
    h, s, v = cv2.split(hsv)
    hue = {"red": (h < 12) | (h > 165), "green": (h > 32) & (h < 85), "blue": (h > 90) & (h < 130)}[
        name
    ]
    mask = (hue & (s > 105) & (v > 20)).astype("uint8")
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        raise ValueError(f"No visible {name} cube in the alignment image")
    label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == label, centers[label]


def compare(view, real, proxy):
    real = cv2.cvtColor(real, cv2.COLOR_BGR2HSV)
    proxy = cv2.cvtColor(proxy, cv2.COLOR_BGR2HSV)
    result = {}
    for name in ("red", "green", "blue"):
        a, ca = cube_mask(real, name)
        b, cb = cube_mask(proxy, name)
        iou = float((a & b).sum() / (a | b).sum())
        error = float(np.linalg.norm(ca - cb))
        result[name] = {
            "iou": iou,
            "centroid_error_px": error,
            "passed": iou >= 0.75 and error <= (8 if view == "head" else 12),
        }
    return result
