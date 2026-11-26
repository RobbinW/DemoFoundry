"""Pick, place and verify a stack in the Marvin workcell."""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import trimesh
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationRootPropertiesCfg,
    DefaultPhysicsCfg,
    JointDrivePropertiesCfg,
    MassPropertiesCfg,
    RigidBodyMaterialCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
    RobotCfg,
)
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg, MeshCollisionCfg
from embodichain.lab.visualization import VisualizationCfg
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from shapely.geometry import MultiPoint

JOINT_NAMES = [f"Joint{i}_{s}" for s in "LR" for i in range(1, 8)]


def pose_matrix(p):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(p[3:7]).as_matrix()
    T[:3, 3] = p[:3]
    return T


def yaw_pose(center, yaw):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    T[:3, 3] = center
    return T


class Kinematics:
    def __init__(self, profile, assets):
        self.reference = np.array(profile["home_q"])
        root = ET.parse(assets / "marvin.urdf").getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        self.chains = {}
        self.bounds = {}
        for side in ["left", "right"]:
            n = "flange_" + side[0].upper()
            chain = []
            limits = []
            while n != "base_link":
                j = by_child[n]
                a = j.find("origin")
                T = np.eye(4)
                if a is not None:
                    T[:3, :3] = Rotation.from_euler(
                        "xyz", np.fromstring(a.get("rpy", "0 0 0"), sep=" ")
                    ).as_matrix()
                    T[:3, 3] = np.fromstring(a.get("xyz", "0 0 0"), sep=" ")
                axis = (
                    np.fromstring(j.find("axis").get("xyz"), sep=" ")
                    if j.find("axis") is not None
                    else np.zeros(3)
                )
                idx = JOINT_NAMES.index(j.get("name")) if j.get("name") in JOINT_NAMES else None
                chain.append((T, axis, idx))
                n = j.find("parent").get("link")
            self.chains[side] = chain[::-1]
            for name in JOINT_NAMES[:7] if side == "left" else JOINT_NAMES[7:]:
                l = root.find(f"joint[@name='{name}']/limit")
                limits.append([float(l.get("lower")) + 0.005, float(l.get("upper")) - 0.005])
            self.bounds[side] = np.array(limits).T

    def fk(self, q, side):
        T = np.eye(4)
        for origin, axis, idx in self.chains[side]:
            T = T @ origin
            if idx is not None:
                A = np.eye(4)
                A[:3, :3] = Rotation.from_rotvec(axis * q[idx]).as_matrix()
                T = T @ A
        return T

    def ik(self, target, q, side):
        ids = np.arange(7) if side == "left" else np.arange(7, 14)
        seed = q[ids].copy()
        lo, hi = self.bounds[side]
        seed = np.clip(seed, lo + 1e-6, hi - 1e-6)

        def residual(x):
            full = q.copy()
            full[ids] = x
            T = self.fk(full, side)
            return np.r_[
                (T[:3, 3] - target[:3, 3]) * 30,
                Rotation.from_matrix(target[:3, :3] @ T[:3, :3].T).as_rotvec() * 3,
                (x - seed) * 0.025,
                (x - self.reference[ids]) * 0.012,
            ]

        sol = least_squares(
            residual,
            seed,
            bounds=(lo, hi),
            max_nfev=70,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        result = q.copy()
        result[ids] = sol.x
        got = self.fk(result, side)
        pe = np.linalg.norm(got[:3, 3] - target[:3, 3])
        ae = Rotation.from_matrix(got[:3, :3] @ target[:3, :3].T).magnitude()
        if pe > 0.0015 or ae > 0.015:
            raise RuntimeError(f"IK failed {side}: position {pe:.4f}m angle {ae:.3f}rad")
        return result


class Skills:
    def __init__(self, scene, out, profile, assets):
        self.scene = scene
        self.motion = scene["motion_limits"]
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.assets = assets
        self.kin = Kinematics(profile, assets)
        self.history = []
        self.events = []
        self.stage = "initial"
        self.held = {}
        self.start = time.monotonic()
        cfg = SimulationManagerCfg(
            headless=True,
            device="cpu",
            physics_cfg=DefaultPhysicsCfg(physics_dt=0.001),
            visualization=VisualizationCfg(backend="none"),
        )
        cfg.render_cfg.renderer = "no-render"
        self.sim = SimulationManager(cfg)
        mat = RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.65, restitution=0.0)

        def box(uid, size, p, yaw=0, kind="static"):
            attrs = (
                RigidBodyPhysicsCfg(material_props=mat, mass_props=MassPropertiesCfg(mass=0.06))
                if kind == "dynamic"
                else RigidBodyPhysicsCfg(material_props=mat)
            )
            return self.sim.add_rigid_object(
                RigidObjectCfg(
                    uid=uid,
                    shape=CubeCfg(size=size),
                    body_type=kind,
                    init_pos=p,
                    init_rot=[0, 0, float(np.rad2deg(yaw))],
                    attrs=attrs,
                )
            )

        table = scene["table"]
        box("table", table["size"], table["center"], table["yaw"])
        t = scene["tray"]
        for part in ["base", "rim"]:
            shape = MeshCfg(
                fpath=str(self.assets / "meshes" / f"tray_{part}.stl"),
                collision=MeshCollisionCfg(
                    approximation="triangle_mesh" if part == "rim" else "convex_hull"
                ),
            )
            self.sim.add_rigid_object(
                RigidObjectCfg(
                    uid="tray_" + part,
                    shape=shape,
                    body_type="static",
                    init_pos=t["center"],
                    init_rot=[0, 0, float(np.rad2deg(t["yaw"]))],
                    attrs=RigidBodyPhysicsCfg(material_props=mat),
                )
            )
        self.blocks = {
            n: box(
                n,
                [b["edge_m"]] * 3,
                (np.array(b["center"]) + [0, 0, 0.001]).tolist(),
                b["yaw_rad"],
                "dynamic",
            )
            for n, b in scene["objects"].items()
        }
        drive = JointDrivePropertiesCfg(
            stiffness={"Joint.*": 6000.0, "slide_.*": 2000.0},
            damping={"Joint.*": 150.0, "slide_.*": 40.0},
            max_effort={"Joint.*": 100.0, "slide_.*": 30.0},
            max_velocity={"Joint.*": 3.0, "slide_.*": 0.2},
            friction=0.0,
        )
        self.robot = self.sim.add_robot(
            RobotCfg(
                uid="marvin",
                fpath=str(self.assets / "marvin.urdf"),
                root_props=ArticulationRootPropertiesCfg(
                    fixed_base=True, self_collision_enabled=False
                ),
                joint_drive_props=drive,
                attrs=RigidBodyPhysicsCfg(material_props=mat),
            )
        )
        self.sim.prepare()
        self.jn = list(self.robot.joint_names)
        self.ln = list(self.robot.link_names)
        self.q = np.array(self.profile["home_q"])
        self.grip = {s: np.array(profile["initial_finger_slides_m"][s]) for s in ("left", "right")}
        self.robot.set_qpos(self.target(), target=False)
        self.robot.set_qpos(self.target(), target=True)
        self.hold(0.5)

    def target(self):
        vals = dict(zip(JOINT_NAMES, self.q))
        for side in self.grip:
            for i, sign in enumerate(["minus", "plus"]):
                vals["slide_" + side + "_" + sign] = float(self.grip[side][i])
        return torch.tensor([[vals[n] for n in self.jn]], dtype=torch.float32)

    def poses(self):
        return {n: b.get_local_pose().cpu().numpy()[0].copy() for n, b in self.blocks.items()}

    def actual_q(self):
        q = self.robot.get_qpos().cpu().numpy()[0]
        return np.array([q[self.jn.index(n)] for n in JOINT_NAMES])

    def tick(self):
        tar = self.target()
        self.history.append(
            {
                "time": len(self.history) * 0.02,
                "q": self.robot.get_qpos().cpu().numpy()[0].copy(),
                "target": tar.numpy()[0].copy(),
                "objects": np.stack(list(self.poses().values())),
                "links": self.robot.body_data.body_link_pose.cpu().numpy()[0].copy(),
                "stage": self.stage,
            }
        )
        self.robot.set_qpos(tar, target=True)
        self.sim.update(step=20)

    def hold(self, seconds):
        for _ in range(round(seconds / 0.02)):
            self.tick()

    def mark(self, name, **data):
        self.stage = name
        rec = {
            "stage": name,
            "sim_s": len(self.history) * 0.02,
            "wall_s": time.monotonic() - self.start,
            **data,
        }
        self.events.append(rec)
        print(json.dumps(rec), flush=True)

    def move_pose(self, side, T, seconds=1.2):
        start = self.kin.fk(self.q, side)
        dist = np.linalg.norm(T[:3, 3] - start[:3, 3])
        angle = Rotation.from_matrix(T[:3, :3] @ start[:3, :3].T).magnitude()
        seconds *= 1.0 if self.stage in ["pick.descend", "place.descend"] else 0.75
        duration = max(
            seconds, dist / self.motion["linear_m_s"], angle / self.motion["angular_rad_s"]
        )
        N = max(12, int(np.ceil(duration / 0.10)))
        slerp = Slerp([0, 1], Rotation.from_matrix([start[:3, :3], T[:3, :3]]))
        path = [self.q.copy()]
        last = self.q.copy()
        for t in np.linspace(0, 1, N + 1)[1:]:
            u = t
            goal = np.eye(4)
            goal[:3, 3] = (1 - u) * start[:3, 3] + u * T[:3, 3]
            goal[:3, :3] = slerp([u]).as_matrix()[0]
            last = self.kin.ik(goal, last, side)
            path.append(last.copy())
        path = np.array(path)
        # Smooth joint curve and quintic progress; scale using velocity/acceleration/jerk.
        curve = CubicSpline(np.linspace(0, 1, len(path)), path, axis=0)
        physical_lower = np.r_[self.kin.bounds["left"][0], self.kin.bounds["right"][0]] - 0.005
        physical_upper = np.r_[self.kin.bounds["left"][1], self.kin.bounds["right"][1]] + 0.005
        shape_preserving = False
        while True:
            steps = int(np.ceil(duration / 0.02))
            t = np.linspace(0, 1, steps + 1)
            u = t**3 * (10 - 15 * t + 6 * t**2)
            samples = curve(u)
            if np.any(samples < physical_lower) or np.any(samples > physical_upper):
                if shape_preserving:
                    raise RuntimeError("IK knots exceed physical joint limits")
                # PCHIP keeps the interpolated path within the IK knot bounds.
                curve = PchipInterpolator(np.linspace(0, 1, len(path)), path, axis=0)
                shape_preserving = True
                continue
            padded = np.vstack(
                [
                    np.repeat(samples[:1], 3, axis=0),
                    samples,
                    np.repeat(samples[-1:], 3, axis=0),
                ]
            )
            velocity = np.diff(padded, axis=0) / 0.02
            accel = np.diff(velocity, axis=0) / 0.02
            jerk = np.diff(accel, axis=0) / 0.02
            scale = max(
                abs(velocity).max() / self.motion["joint_velocity_rad_s"],
                np.sqrt(abs(accel).max() / self.motion["joint_acceleration_rad_s2"]),
                np.cbrt(abs(jerk).max() / self.motion["joint_jerk_rad_s3"]),
            )
            if scale <= 1.001:
                break
            duration *= scale * 1.03
        if shape_preserving:
            self.events.append(
                {
                    "stage": "motion.interpolation_fallback",
                    "sim_s": len(self.history) * 0.02,
                    "arm": side,
                    "interpolator": "PCHIP",
                    "reason": "cubic target exceeded physical URDF limit",
                }
            )
        for sample in samples[1:]:
            self.q = sample
            self.tick()
        self.hold(0.10)

    def move_up(self, side, dz=0.10):
        T = self.kin.fk(self.q, side)
        T[2, 3] += dz
        self.move_pose(side, T)

    def gripper(self, side, closed):
        a = self.grip[side].copy()
        b = np.array(self.profile["grasps"][side]["close_target_m"]) if closed else np.zeros(2)
        if np.max(abs(a - b)) < 1e-7:
            return
        for u in np.linspace(0, 1, 41)[1:]:
            self.grip[side] = a + (b - a) * u
            self.tick()
        self.hold(0.4)

    def choose_grasp(self, obj, side):
        p = self.poses()[obj]
        yaw = float(Rotation.from_quat(p[3:]).as_euler("xyz")[2])
        cal = np.array(self.profile["grasps"][side]["T_object_flange"])
        seed = np.array(self.profile["grasps"][side]["ik_seed"])
        options = []
        for symmetry in range(4):
            T = yaw_pose(p[:3], yaw + symmetry * np.pi / 2) @ cal
            # Grip higher on the cube so fingertips clear the lower cube during stacking.
            T[2, 3] += 0.008
            off = T[:3, 3] - p[:3]
            # Both arm bases are behind the table along the world X axis.
            if off[0] > -0.06:
                continue
            pre = T.copy()
            pre[2, 3] += 0.10
            try:
                q = self.kin.ik(pre, seed, side)
                self.kin.ik(T, q, side)
                score = np.linalg.norm(q - (self.q)) + 0.5 * abs(off[0])
                options.append((score, T, symmetry))
            except RuntimeError:
                continue
        if not options:
            raise RuntimeError(f"no reachable grasp for {obj} using {side}")
        return min(options, key=lambda x: x[0])[1:]

    def pick(self, obj, arm="auto"):
        side = ("left" if self.poses()[obj][1] > -0.06 else "right") if arm == "auto" else arm
        self.mark("pick.approach", object=obj, arm=side)
        self.gripper(side, False)
        T, sym = self.choose_grasp(obj, side)
        self.move_up(side, 0.10)
        pre = T.copy()
        pre[2, 3] += 0.10
        self.move_pose(side, pre)
        self.mark("pick.descend", object=obj, arm=side, symmetry=sym)
        self.move_pose(side, T, 1.2)
        self.mark("pick.close", object=obj, arm=side)
        z0 = self.poses()[obj][2]
        self.gripper(side, True)
        self.mark("pick.lift", object=obj, arm=side)
        self.move_up(side, 0.11)
        self.hold(0.3)
        p = self.poses()[obj]
        lift = float(p[2] - z0)
        if lift < 0.06:
            raise RuntimeError(f"grasp did not lift {obj}: {lift:.3f}m")
        tf = self.kin.fk(self.actual_q(), side)
        self.held[side] = {
            "object": obj,
            "T_flange_object": np.linalg.inv(tf) @ pose_matrix(p),
        }
        self.mark("pick.verified", object=obj, arm=side, lift_m=lift)
        return side

    def place_on(self, obj, support, arm):
        held = self.held[arm]
        if held["object"] != obj:
            raise ValueError(f"{arm} holds {held['object']}, expected {obj}")
        sp = self.poses()[support]
        edge = self.scene["objects"][obj]["edge_m"]
        low = self.scene["objects"][support]["edge_m"]
        p = self.poses()[obj]
        yaw = float(Rotation.from_quat(p[3:]).as_euler("xyz")[2])
        support_yaw = float(Rotation.from_quat(sp[3:]).as_euler("xyz")[2])
        yaw = support_yaw + round((yaw - support_yaw) / (np.pi / 2)) * (np.pi / 2)
        desired = yaw_pose([sp[0], sp[1], sp[2] + (edge + low) / 2 + 0.0008], yaw)
        T = desired @ np.linalg.inv(held["T_flange_object"])
        pre = T.copy()
        pre[2, 3] += 0.10
        self.mark("place.transport", object=obj, support=support, arm=arm)
        self.move_pose(arm, pre, 1.8)
        self.mark("place.descend", object=obj, support=support, arm=arm)
        self.move_pose(arm, T, 1.5)
        self.hold(0.3)
        self.mark("place.release", object=obj, arm=arm)
        self.gripper(arm, False)
        self.held.pop(arm)
        self.move_up(arm, 0.12)
        self.hold(0.7)
        support_metrics = self.support(obj, support)
        self.mark(
            "place.verified",
            object=obj,
            support=support,
            metrics=support_metrics,
        )
        if not support_metrics["passed"]:
            raise RuntimeError(f"placement failed: {obj} on {support}")

    def park(self, arm):
        self.mark("park", arm=arm)
        goal = self.kin.fk(np.array(self.profile["home_q"]), arm)
        goal[2, 3] += 0.12
        self.move_pose(arm, goal, 2.0)

    def support(self, upper, lower):
        poses = self.poses()
        up = poses[upper]
        lo = poses[lower]

        def face(name, p, top):
            e = self.scene["objects"][name]["edge_m"]
            R = Rotation.from_quat(p[3:]).as_matrix()
            a = int(np.argmax(abs(R[2])))
            sign = np.sign(R[2, a]) * (1 if top else -1)
            v = np.array(
                [
                    [x, y, z]
                    for x in [-e / 2, e / 2]
                    for y in [-e / 2, e / 2]
                    for z in [-e / 2, e / 2]
                ]
            )
            pts = v[np.isclose(v[:, a], sign * e / 2)] @ R.T + p[:3]
            return pts, float(np.arccos(np.clip(abs(R[2, a]), 0, 1)) * 180 / np.pi)

        uf, ut = face(upper, up, False)
        lf, lt = face(lower, lo, True)
        # Measure overlap between the two projected support faces.
        uh = MultiPoint(uf[:, :2]).convex_hull
        lh = MultiPoint(lf[:, :2]).convex_hull
        ratio = float(uh.intersection(lh).area / uh.area)
        gap = float(uf[:, 2].mean() - lf[:, 2].mean())
        return {
            "overlap": ratio,
            "gap_m": gap,
            "tilt_deg": max(ut, lt),
            "xy_m": float(np.linalg.norm(up[:2] - lo[:2])),
            "passed": bool(ratio > 0.65 and abs(gap) < 0.0015 and max(ut, lt) < 5),
        }

    def finish(self):
        self.mark("verify.final")
        begin = len(self.history)
        self.hold(5.0)
        held = np.stack([r["objects"] for r in self.history[begin:]])
        drift = float(np.linalg.norm(np.ptp(held[:, :, :3], axis=0), axis=1).max())
        rotation_span = 0.0
        for k in range(len(self.blocks)):
            rr = Rotation.from_quat(held[:, k, 3:]).as_matrix()
            rotation_span = max(
                rotation_span,
                float(np.rad2deg(Rotation.from_matrix(rr @ rr[-1].T).magnitude()).max()),
            )
        # Bounding-box separation gives a lower bound on gripper clearance.
        block_bounds = []
        for n, p in self.poses().items():
            e = self.scene["objects"][n]["edge_m"] / 2
            v = np.array([[x, y, z] for x in [-e, e] for y in [-e, e] for z in [-e, e]])
            v = v @ Rotation.from_quat(p[3:]).as_matrix().T + p[:3]
            block_bounds.append((v.min(0), v.max(0)))
        linkposes = self.robot.body_data.body_link_pose.cpu().numpy()[0]
        clearances = []
        for link in ET.parse(self.assets / "marvin.urdf").getroot().findall("link"):
            if not link.get("name").startswith("gripper_"):
                continue
            T = pose_matrix(linkposes[self.ln.index(link.get("name"))])
            for visual in link.findall("visual"):
                mesh = trimesh.load(self.assets / visual.find("geometry/mesh").get("filename"))
                v = mesh.vertices @ T[:3, :3].T + T[:3, 3]
                mn, mx = v.min(0), v.max(0)
                for bn, bx in block_bounds:
                    clearances.append(
                        float(np.linalg.norm(np.maximum(0, np.maximum(mn - bx, bn - mx))))
                    )
        clearance = min(clearances)
        supports = {
            f"{u}_on_{l}": self.support(u, l) for u, l in [("red", "green"), ("blue", "red")]
        }
        opened = all(np.max(abs(g)) < 1e-8 for g in self.grip.values())
        return {
            "success": bool(
                all(s["passed"] for s in supports.values())
                and drift < 0.001
                and rotation_span < 0.5
                and clearance > 0.05
                and opened
                and not self.held
            ),
            "supports": supports,
            "hold_5s_drift_m": drift,
            "hold_5s_rotation_span_deg": rotation_span,
            "gripper_clearance_lower_bound_m": clearance,
            "grippers_commanded_open": opened,
            "observation_source": "simulator state at skill boundaries",
            "control_hz": 50,
            "duration_s": len(self.history) * 0.02,
            "wall_seconds": time.monotonic() - self.start,
        }

    def save(self, result):
        h = self.history
        np.savez_compressed(
            self.out / "states.npz",
            time_s=[r["time"] for r in h],
            q_actual=[r["q"] for r in h],
            q_target=[r["target"] for r in h],
            object_poses=[r["objects"] for r in h],
            link_poses=[r["links"] for r in h],
            stage=[r["stage"] for r in h],
            joint_names=self.jn,
            link_names=self.ln,
            object_names=list(self.blocks),
        )
        (self.out / "events.json").write_text(json.dumps(self.events, indent=2))
        (self.out / "result.json").write_text(json.dumps(result, indent=2))
        (self.out / "scene.json").write_text(json.dumps(self.scene, indent=2))
        self.sim.destroy(exit_process=False)
