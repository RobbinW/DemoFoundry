"""Fit the reference layout, execute the two-arm program and assess its trajectory."""

import json
from pathlib import Path

import cv2

from .assess_motion import assess
from .first_frame import reconstruct
from .program import INSTRUCTION, stack_from_real_layout
from .runtime import JOINT_NAMES, Skills


def simulate(workcell: Path, run: Path):
    profile = json.loads((workcell / "config/robot.json").read_text())
    cameras = json.loads((workcell / "config/cameras.json").read_text())
    if profile["joint_names"] != JOINT_NAMES:
        raise ValueError("home_q must follow Joint1_L … Joint7_L, then Joint1_R … Joint7_R")
    for view, parent in (("head", "base_link"), ("left", "flange_L"), ("right", "flange_R")):
        camera = cameras[view]
        image = cv2.imread(str(workcell / "references" / f"{view}_initial.png"))
        if image is None or image.shape[:2] != (camera["height"], camera["width"]):
            raise ValueError(f"{view} reference must have the calibrated image dimensions")
        if camera["parent"] != parent:
            raise ValueError(f"The {view} transform must be expressed in {parent}")

    scene = reconstruct(
        workcell / "references/head_initial.png", run / "fit", profile, cameras["head"]
    )
    scene["instruction"] = INSTRUCTION
    scene["table"] = profile["table"]
    scene["tray"] = profile["tray"]
    scene["motion_limits"] = {
        "linear_m_s": 0.24,
        "angular_rad_s": 1.2,
        "joint_velocity_rad_s": 1.3,
        "joint_acceleration_rad_s2": 4.0,
        "joint_jerk_rad_s3": 32.0,
    }
    assets = workcell / "robot"
    skills = Skills(scene, run, profile, assets)
    try:
        result = stack_from_real_layout(skills)
    except Exception as exc:
        skills.save({"success": False, "stage": skills.stage, "error": str(exc)})
        raise
    skills.save(result)
    metrics = assess(run, assets / "marvin.urdf")
    if not result["success"] or not metrics["trajectory_passed"]:
        raise RuntimeError(
            f"Stack or trajectory checks failed. See {run}/result.json and motion_metrics.json"
        )
