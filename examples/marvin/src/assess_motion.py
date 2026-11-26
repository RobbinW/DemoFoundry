"""Measure joint limits, target derivatives and pick order."""

import json
import xml.etree.ElementTree as ET

import numpy as np


def joint_limit_metrics(p, urdf):
    root = ET.parse(urdf).getroot()
    names = [f"Joint{i}_{s}" for s in "LR" for i in range(1, 8)]
    bounds = np.array(
        [
            [float(root.find(f"joint[@name='{n}']/limit").get(k)) for k in ["lower", "upper"]]
            for n in names
        ]
    )
    ids = [list(p["joint_names"]).index(n) for n in names]
    metrics = {}
    for key in ["q_target", "q_actual"]:
        q = p[key][:, ids]
        metrics[key + "_minimum_margin_rad"] = float(
            np.minimum(q - bounds[:, 0], bounds[:, 1] - q).min()
        )
    metrics["passed"] = (
        metrics["q_target_minimum_margin_rad"] >= 0
        and metrics["q_actual_minimum_margin_rad"] >= -1e-4
    )
    metrics["actual_solver_tolerance_rad"] = 1e-4
    return metrics


def assess(run, urdf):
    p = np.load(run / "states.npz")
    events = json.loads((run / "events.json").read_text())
    names = list(p["joint_names"])
    ids = [i for i, n in enumerate(names) if n.startswith("Joint")]
    out = {
        "episode": run.name,
        "duration_s": float(p["time_s"][-1] + 0.02),
        "pick_order": [e["object"] for e in events if e["stage"] == "pick.approach"],
    }
    for key in ["q_target", "q_actual"]:
        a = p[key][:, ids]
        v = np.diff(a, axis=0) / 0.02
        acc = np.diff(v, axis=0) / 0.02
        j = np.diff(acc, axis=0) / 0.02
        out[key] = {
            "max_velocity_rad_s": float(abs(v).max()),
            "max_acceleration_rad_s2": float(abs(acc).max()),
            "max_jerk_rad_s3": float(abs(j).max()),
            "p99_jerk_rad_s3": float(np.quantile(abs(j), 0.99)),
            "total_joint_variation_rad": float(abs(np.diff(a, axis=0)).sum()),
        }
    out["max_tracking_error_rad"] = float(abs(p["q_target"][:, ids] - p["q_actual"][:, ids]).max())
    out["joint_limits"] = joint_limit_metrics(p, urdf)
    limits = json.loads((run / "scene.json").read_text())["motion_limits"]
    out["motion_limits"] = limits
    out["trajectory_passed"] = bool(
        out["q_target"]["max_velocity_rad_s"] <= limits["joint_velocity_rad_s"] * 1.01
        and out["q_target"]["max_acceleration_rad_s2"] <= limits["joint_acceleration_rad_s2"] * 1.01
        and out["q_target"]["max_jerk_rad_s3"] <= limits["joint_jerk_rad_s3"] * 1.02
        and out["pick_order"] == ["red", "blue"]
        and out["joint_limits"]["passed"]
    )
    (run / "motion_metrics.json").write_text(json.dumps(out, indent=2))
    return out
