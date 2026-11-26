"""Export calibrated H.264 proxies and measured joint states with their target commands."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from .alignment import compare


def proxy_frames(workcell, run, view, camera, count):
    width, height = camera["width"], camera["height"]
    intrinsic = np.array(camera["K"]).reshape(3, 3)
    y, x = np.indices((height, width), dtype="float32")
    mapping = cv2.undistortPoints(
        np.dstack((x, y)).reshape(-1, 1, 2), intrinsic, np.array(camera["D"]), P=intrinsic
    ).reshape(height, width, 2)
    print(f"Rendering {view} · {count} frames", flush=True)
    with TemporaryDirectory(dir=run) as folder:
        subprocess.run(
            [
                os.environ.get("BLENDER", "blender"),
                "-q",
                "-b",
                str(workcell / "scene.blend"),
                "--python-exit-code",
                "1",
                "--python",
                str(Path(__file__).with_name("proxy.py")),
                "--",
                str(workcell),
                str(run),
                view,
                folder,
            ],
            check=True,
        )
        for index in range(count):
            path = Path(folder) / f"{index:05d}.png"
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f"Missing rendered frame {path.name} for {view}")
            yield cv2.remap(frame, mapping, None, cv2.INTER_LINEAR)


def write_video(path, camera, frames):
    size = f"{camera['width']}x{camera['height']}"
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        size,
        "-r",
        "25",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
        try:
            for image in frames:
                encoder.stdin.write(image.tobytes())
        finally:
            encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError(f"FFmpeg failed for {path.name}")


def joint_channels(states, field):
    names = states["joint_names"].tolist()
    joints = states[field]
    fields = []
    order = []
    for side in ("left", "right"):
        arm = [f"Joint{i}_{side[0].upper()}" for i in range(1, 8)]
        fingers = [names.index(f"slide_{side}_{sign}") for sign in ("minus", "plus")]
        fields.extend(
            [
                joints[:, [names.index(n) for n in arm]],
                joints[:, fingers].mean(axis=1, keepdims=True) / 0.043,
            ]
        )
        order.extend([*arm, f"{side}_gripper_closure"])
    return np.hstack(fields).astype("float32"), order


def export(workcell, run):
    result = json.loads((run / "result.json").read_text())
    metrics = json.loads((run / "motion_metrics.json").read_text())
    if not (result["success"] and metrics["trajectory_passed"]):
        raise ValueError("The simulated episode failed its task or motion checks")
    cameras = json.loads((workcell / "config/cameras.json").read_text())
    scene = json.loads((run / "scene.json").read_text())
    with np.load(run / "states.npz") as data:
        states = dict(data)
    times = states["time_s"]
    indices = np.arange(0, len(times), 2)
    np.testing.assert_allclose(times, np.arange(len(times)) / 50, rtol=0, atol=1e-8)
    actions, order = joint_channels(states, "q_target")
    measured, _ = joint_channels(states, "q_actual")
    np.testing.assert_equal(np.isfinite(actions).all() and np.isfinite(measured).all(), True)
    (run / "proxy").mkdir(exist_ok=True)
    references = run / "references"
    references.mkdir(exist_ok=True)
    counts = {}
    alignment = {}
    for view, camera in cameras.items():
        ref = workcell / "references" / f"{view}_initial.png"
        shutil.copyfile(ref, references / f"{view}.png")
        path = run / "proxy" / f"{view}.mp4"
        write_video(path, camera, proxy_frames(workcell, run, view, camera, len(indices)))
        video = cv2.VideoCapture(str(path))
        count = 0
        try:
            np.testing.assert_allclose(video.get(cv2.CAP_PROP_FPS), 25, atol=1e-4)
            while True:
                ok, frame = video.read()
                if not ok:
                    break
                np.testing.assert_equal(frame.shape[:2], (camera["height"], camera["width"]))
                if count == 0:
                    real = cv2.imread(str(ref))
                    alignment[view] = compare(view, real, frame)
                    cv2.imwrite(
                        str(run / "fit" / f"{view}_comparison.jpg"), np.hstack([real, frame])
                    )
                count += 1
        finally:
            video.release()
        np.testing.assert_equal(count, len(indices))
        counts[view] = count

    np.savez_compressed(
        run / "episode.npz",
        control_time_s=times,
        state=measured,
        action=actions,
        frame_time_s=times[indices],
        frame_state_index=indices,
    )
    metadata = {
        "instruction": scene["instruction"],
        "dimension_order": order,
        "action": "absolute arm position targets (rad), mean finger closure / 0.043 m",
        "state": "measured joint positions in the same units immediately before each action",
        "control_hz": 50,
        "proxy_fps": 25,
        "cameras": cameras,
    }
    (run / "metadata.json").write_text(json.dumps(metadata, indent=2))
    aligned = all(cube["passed"] for view in alignment.values() for cube in view.values())
    report = {
        "passed": aligned,
        "alignment": alignment,
        "control_steps": len(times),
        "decoded_frames": counts,
        "duration_s": result["duration_s"],
        "supports": result["supports"],
    }
    (run / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not aligned:
        raise ValueError(
            "First-frame alignment failed. Inspect fit/*_comparison.jpg and validation.json"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workcell", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    export(args.workcell, args.run)
