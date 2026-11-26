"""Fit three cube poses to their silhouettes in the head-camera image."""

import hashlib
import itertools
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def fit_cube(edge, contour, initial, z, C, K, D):
    """Fit XY and yaw by matching the projected convex hull to the observed contour."""
    base = np.array(list(itertools.product([-0.5, 0.5], repeat=3)))

    def vertices(x):
        return base * edge @ Rotation.from_euler("z", x[2]).as_matrix().T + [
            x[0],
            x[1],
            z,
        ]

    def project(x):
        pts = vertices(x) @ C[:3, :3].T + C[:3, 3]
        return cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, D)[0][:, 0]

    def residual(x):
        poly = cv2.convexHull(project(x).astype("float32")).reshape(-1, 2).astype(float)
        ab = np.roll(poly, -1, axis=0) - poly
        u = np.clip(
            np.sum((contour[:, None] - poly) * ab, axis=-1)
            / np.maximum(np.sum(ab * ab, axis=-1), 1e-9),
            0,
            1,
        )
        d = np.linalg.norm(contour[:, None] - (poly + u[:, :, None] * ab), axis=-1).min(1)
        samples = np.concatenate([a + b * np.linspace(0, 1, 10)[:, None] for a, b in zip(poly, ab)])
        return np.r_[d, np.linalg.norm(samples[:, None] - contour, axis=-1).min(1)]

    def jac(x):
        eps = [1e-5, 1e-5, 1e-4]
        return np.column_stack(
            [(residual(x + np.eye(3)[j] * eps[j]) - residual(x)) / eps[j] for j in range(3)]
        )

    fits = [
        least_squares(
            residual,
            [*initial[:2], yaw],
            jac=jac,
            bounds=(
                [initial[0] - 0.1, initial[1] - 0.1, -np.pi],
                [initial[0] + 0.1, initial[1] + 0.1, np.pi],
            ),
            loss="soft_l1",
            f_scale=2,
            max_nfev=100,
        )
        for yaw in [-0.7, 0, 0.7]
    ]
    f = min(fits, key=lambda f: np.mean(residual(f.x) ** 2))
    poly = np.round(cv2.convexHull(project(f.x).astype("float32"))).astype("int32")
    return f.x, poly, float(np.sqrt(np.mean(residual(f.x) ** 2)))


def reconstruct(image_path: Path, out: Path, profile, camera):
    K = np.array(camera["K"]).reshape(3, 3)
    D = np.array(camera["D"])
    T = np.array(camera["T_parent_camera"])
    C = np.linalg.inv(T)
    im = cv2.imread(str(image_path))
    if im is None or im.shape[:2] != (camera["height"], camera["width"]):
        raise ValueError("Head reference must match its calibrated resolution")
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    result = {
        "input_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "input_image": image_path.name,
        "objects": {},
        "method": "Calibrated head RGB silhouette fit with known cube sizes and support heights",
    }
    out.mkdir(parents=True, exist_ok=True)
    overlay = im.copy()
    for name, edge in profile["sizes"].items():
        mask = (
            (((h < 12) | (h > 165)) & (s > 120) & (v > 65))
            if name == "red"
            else (
                ((h > 32) & (h < 85) & (s > 105) & (v > 60))
                if name == "green"
                else ((h > 90) & (h < 130) & (s > 110) & (v > 55))
            )
        )
        mask[: int(im.shape[0] * 0.35)] = False
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"))
        if n < 2:
            raise ValueError(f"Missing {name} cube in the head reference")
        mask = (lab == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype("uint8") * 255
        if cv2.countNonZero(mask) <= 60:
            raise ValueError(f"Too few pixels to fit the {name} cube")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        p = max(contours, key=cv2.contourArea)[:, 0].astype(float)[::2]
        mu = cv2.moments(mask)
        uv = np.array([mu["m10"] / mu["m00"], mu["m01"] / mu["m00"]])
        ray = T[:3, :3] @ np.r_[cv2.undistortPoints(uv.reshape(1, 1, 2), K, D)[0, 0], 1]
        # Intersect the camera ray with the table or tray support plane.
        z = profile["table"]["support_z_m"] + edge / 2
        estimate = T[:3, 3] + ray * ((z - T[2, 3]) / ray[2])
        tray = profile["tray"]
        tray_local = Rotation.from_euler("z", -tray["yaw"]).apply(
            estimate - np.array(tray["center"])
        )
        on_tray = bool(np.all(abs(tray_local[:2]) < np.array(tray["size"][:2]) / 2))
        z += tray["size"][2] if on_tray else 0
        initial = T[:3, 3] + ray * ((z - T[2, 3]) / ray[2])

        pose, poly, rms = fit_cube(edge, p, initial, z, C, K, D)
        pred = np.zeros(mask.shape, np.uint8)
        cv2.fillConvexPoly(pred, poly, 255)
        iou = float(np.sum((pred > 0) & (mask > 0)) / np.sum((pred > 0) | (mask > 0)))
        result["objects"][name] = {
            "center": [float(pose[0]), float(pose[1]), z],
            "edge_m": edge,
            "yaw_rad": float(pose[2]),
            "support": "tray" if on_tray else "table",
            "fit_silhouette_iou": iou,
            "fit_edge_rms_px": rms,
        }
        cv2.polylines(overlay, [poly], True, (0, 255, 255), 1)
        cv2.imwrite(str(out / f"mask_{name}.png"), mask)
    cv2.imwrite(str(out / "first_frame_overlay.jpg"), overlay)
    return result
