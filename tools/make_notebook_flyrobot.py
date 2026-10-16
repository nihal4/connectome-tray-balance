"""Generates notebooks/kaggle_flyrobot.ipynb. Edit here, then run:
    python tools/make_notebook_flyrobot.py
"""
import json
import os

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# Fly-robot: a robot arm run by a piece of the real fly brain

The robot's camera is wired into the fly's **eye** (photoreceptors), its joints into the fly's
**body-sense** neurons (ascending neurons), and its motors are driven by the fly's **command**
neurons (descending neurons). In between: ~2,500 neurons on the real pathways through the
optic lobe and central brain (FlyWire connectome). Only the fly brain is trained (PPO).

The brain sees the tray through a 16x16 wrist camera (this frame + the previous one). It is
**not** told where the ball is, where the walls are or where the goal is.

**Tasks (`ENV`):**
- `tray_vision`: keep the ball on the tray (solved: ~250k steps on Kaggle)
- `maze_fixed3_vision`: one fixed 3x3 tilt maze, seen only through the camera  <- current
- `maze3_vision`: a new random maze every episode
- `maze3h_vision`: random mazes with holes

**Kaggle settings:** Internet ON. Accelerator: **None (CPU)**. About 2.7 h per 1M steps.
Use **Save & Run All (Commit)** for long runs. Checkpoints are saved every 50k steps.
""")
code("""
# ===== Settings =====
REPO_URL = "https://github.com/nihal4/connectome-tray-balance.git"
BRANCH = "main"

ENV = "maze_fixed3_vision"   # tray_vision -> maze_fixed3_vision -> maze3_vision -> maze3h_vision
TOTAL_STEPS = 2_000_000      # mazes need more than the tray (~5.5 h at this size)
N_NEURONS = 2500             # brain size (eye 512 + body 64 + commands 64 + pathway neurons)
RANDOM_START = 0.5           # maze curriculum: share of training episodes that start the ball at a random
                             # point on the route (evaluation always uses the real start). 0 = off
SEED = 0
DEVICE = "cpu"
""")
code("""
import os, subprocess, sys, glob, shutil
WORK = "/kaggle/working"
REPO = f"{WORK}/connectome-tray-balance"
if os.path.exists(REPO):
    subprocess.run(["git", "-C", REPO, "pull"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, REPO], check=True)
os.chdir(REPO)
sys.path.insert(0, REPO)
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
""")
code("""
!pip install -q mujoco gymnasium stable-baselines3 psutil imageio imageio-ffmpeg
""")
md("## 1. Download the connectome + cell types, build the fly-robot brain")
code("""
!python scripts/download_connectome.py --out data/raw
!python scripts/build_flybrain.py --n {N_NEURONS}
BRAIN = sorted(glob.glob(f"data/flybrain_cam16_N*.npz"))[-1]
print("brain file:", BRAIN)
""")
md("""
`eye_to_command_hops` shows how many synapses a signal crosses from the eye to the command
neurons inside this brain (3-5 for all of them).

## 2. Check the task (a hand-written controller can solve it, doing nothing cannot)
""")
code("""
!python scripts/check_mac.py --env {ENV} --subset {BRAIN}
""")
md("## 3. Train the fly brain")
code("""
import re
KEEP = re.compile(r"^\\[|Eval num|Episode length|Success rate|New best")
def train(env=ENV, extra=""):
    cmd = (f"{sys.executable} -u scripts/train.py --env {env} --policy connectome --subset {BRAIN} "
           f"--profile kaggle --total_steps {TOTAL_STEPS} --seed {SEED} --device {DEVICE} "
           f"--random_start {RANDOM_START if ENV.startswith('maze') else 0} --runs_dir {WORK}/runs {extra}")
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if KEEP.search(line):
            print(line, end="", flush=True)
    if p.wait() != 0:
        print(f"!! exited with code {p.returncode}")

train()
""")
md("""
To continue a run (after a timeout, or for more steps): raise `TOTAL_STEPS` and run
`train(extra="--resume")`. It continues from the last checkpoint, with the current
`RANDOM_START` setting, so an existing maze run can be switched to the curriculum this way.

## 4. Results
""")
code("""
run = f"{WORK}/runs/{ENV}-connectome-real-s{SEED}"
!python scripts/plot_runs.py {run} --out {WORK}/curves_{ENV}.png
from IPython.display import Image, Video, display
display(Image(f"{WORK}/curves_{ENV}.png"))
""")
code("""
# Tests it never saw in training: camera noise, blind (ball invisible), heavy ball, pushes
# (+ unseen mazes for maze_fixed3_vision)
!python scripts/evaluate.py --run {run} --robustness --episodes 10
""")
md("""
## 5. Videos

One video per test. The camera image the brain receives is in the top-left corner.
Heavy ball: drawn dark grey. Pushes: the frame turns red on each shove.
""")
code("""
import numpy as np, imageio.v2 as imageio
os.environ["MUJOCO_GL"] = "egl"
from scripts.evaluate import load, make_norm, run_config, add_eye_view
from ctb.envs import build_env

env_name = run_config(run)["env"]
TESTS = {"clean": {}, "heavy_ball": {"ball_mass_scale": 3.0}, "push": {"push_prob": 0.05}}
if env_name.startswith("maze_fixed3"):
    TESTS["unseen_maze"] = {"fixed_maze": False}

model = load(run)
vn = make_norm(run, {})          # same input normalisation as in training

def record_test(name, kw, seed=7):
    env = build_env(env_name, seed, render_mode="rgb_array", **kw)
    if name == "heavy_ball":
        env.model.geom_rgba[env.model.geom("ball").id] = (0.25, 0.25, 0.25, 1)
    obs, _ = env.reset(seed=seed)
    frames, pushes, info = [], 0, {}
    for t in range(env.max_steps):
        a, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
        obs, _, term, trunc, info = env.step(a[0])
        frame = env.render()
        if getattr(env, "vision", False):
            frame = add_eye_view(frame, env.camera_frame())
        if np.any(env.data.xfrc_applied[env.ball_body] != 0):
            pushes += 1
            frame = frame.copy(); frame[:8] = frame[-8:] = frame[:, :8] = frame[:, -8:] = (220, 40, 40)
        frames.append(frame)
        if term or trunc:
            break
    frames += [frames[-1]] * 25
    path = f"{WORK}/{env_name}_{name}.mp4"
    imageio.mimsave(path, frames, fps=50)
    env.close()
    solved = info.get("is_success", (not info.get("fell", False)) and t + 1 >= env.max_steps)
    print(f"{name:12s} {t + 1}/{env.max_steps} steps, solved={solved}, pushes={pushes} -> {path}")
    return path

for name, kw in TESTS.items():
    display(Video(record_test(name, kw), embed=True, width=420))
""")
md("## 6. Export everything (runs, videos, curves) as one zip")
code("""
import zipfile
out = f"{WORK}/runs_export.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{WORK}/runs/**/*", recursive=True):
        if os.path.isfile(f):
            z.write(f, os.path.relpath(f, WORK))
    for f in glob.glob(f"{WORK}/*.mp4") + glob.glob(f"{WORK}/*.png"):
        z.write(f, os.path.join("media", os.path.basename(f)))
with zipfile.ZipFile(out) as z:
    names = z.namelist()
print(f"saved {out} ({os.path.getsize(out) / 1e6:.1f} MB): "
      f"{sum(n.startswith('runs/') for n in names)} run files, "
      f"{sum(n.startswith('media/') for n in names)} videos/images")
print("download runs_export.zip from the Output panel")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    src = c["source"].split("\n")
    c["source"] = [line + "\n" for line in src[:-1]] + [src[-1]]
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
json.dump(nb, open(os.path.join(root, "notebooks", "kaggle_flyrobot.ipynb"), "w"), indent=1)
print("wrote notebooks/kaggle_flyrobot.ipynb")
