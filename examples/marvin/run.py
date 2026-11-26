"""Generate one calibrated Marvin stacking demonstration."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workcell", type=Path, default=HERE / "workcell")
    parser.add_argument("--out", type=Path, required=True, help="a new episode directory")
    args = parser.parse_args()
    workcell = args.workcell.resolve()
    run = args.out.resolve()
    if shutil.which("ffmpeg") is None:
        parser.error("Install FFmpeg with libx264 on PATH")
    if shutil.which(os.environ.get("BLENDER", "blender")) is None:
        parser.error("Install Blender 4.5 LTS or set BLENDER to its executable")
    run.mkdir(parents=True, exist_ok=False)
    from src.simulate import simulate

    simulate(workcell, run)
    subprocess.run(
        [sys.executable, "-m", "src.export_episode", str(workcell), str(run)], cwd=HERE, check=True
    )


if __name__ == "__main__":
    main()
