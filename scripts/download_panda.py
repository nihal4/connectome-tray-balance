"""Download the Franka Emika Panda robot model (arm + 2-finger gripper) from
MuJoCo Menagerie (google-deepmind/mujoco_menagerie, Apache-2.0 licence) into
data/menagerie/franka_emika_panda. About 35 MB of meshes; needs git.

    python scripts/download_panda.py
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "menagerie")
URL = "https://github.com/google-deepmind/mujoco_menagerie.git"


def main():
    target = os.path.join(DEST, "franka_emika_panda", "panda.xml")
    if os.path.exists(target):
        print(f"[skip] {target} already exists")
        return
    tmp = os.path.join(DEST, "_clone")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(DEST, exist_ok=True)
    run = lambda *a: subprocess.run(a, check=True)  # noqa: E731
    run("git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", URL, tmp)
    run("git", "-C", tmp, "sparse-checkout", "set", "franka_emika_panda")
    shutil.move(os.path.join(tmp, "franka_emika_panda"), os.path.join(DEST, "franka_emika_panda"))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[ok] Panda model in {os.path.dirname(target)}")


if __name__ == "__main__":
    sys.exit(main())
