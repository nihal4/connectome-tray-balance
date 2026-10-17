"""Generates notebooks/kaggle_drone.ipynb. Edit here, then run:
    python tools/make_notebook_drone.py
"""
import json
import os

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# Fly-brain drone: a quadcopter steered by a piece of the real fly brain

The drone sees through a **fly-like compound eye**: two eyes, 160 facets each, almost
all-round vision, every facet blurry like a real one. Each facet drives its own
photoreceptors in the real fly wiring (FlyWire): **R1-6** (brightness, used for motion),
**R7** (ultraviolet) and **R8** (green). Its body sense (airspeed, tilt, rotation) goes into the
fly's **ascending** neurons, and its flight commands come out of the fly's **descending**
(command) neurons. In between: ~4,000 neurons on the real pathways. Only the brain is trained.

A simple stabiliser keeps the drone level, like the fly's halteres (balance organs).
The brain decides where to fly: forward/back, left/right, up/down, turn.

**Task `drone_reach`:** a yellow ball (25 cm) appears somewhere new in the room every episode (start
position and heading random too). Find it with the eyes, fly to it and hover on it for 0.4 s.
The brain is never told where the ball is. Training already includes light wind and gusts,
eye noise, a slightly different drone weight each episode and up to two blue decoy balls.

**Kaggle settings:** Internet ON. Accelerator: **None (CPU)**. About 1.5-2 h per 1M steps.
Use **Save & Run All (Commit)**. Checkpoints are saved every 50k steps.
""")
code("""
# ===== Settings =====
REPO_URL = "https://github.com/nihal4/connectome-tray-balance.git"
BRANCH = "main"

ENV = "drone_reach"
TOTAL_STEPS = 2_000_000
N_NEURONS = 4000             # brain size (eye 960 + body 64 + commands 64 + pathway neurons)
FACETS = 160                 # facets per eye
RANDOM_START = 0.3           # share of training episodes that start near the ball (evaluation never does)
SEED = 0
DEVICE = "cpu"
RUN_NAME = f"{ENV}-connectome-real-s{SEED}"
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
!pip install -q mujoco gymnasium stable-baselines3 psutil imageio imageio-ffmpeg scipy
""")
md("## 1. Download the connectome + cell types, build the drone brain")
code("""
!python scripts/download_connectome.py --out data/raw
!python scripts/build_flybrain.py --eye compound --facets {FACETS} --n {N_NEURONS} --hops 5
BRAIN = sorted(glob.glob(f"data/flybrain_eye{FACETS}_N*.npz"))[-1]
print("brain file:", BRAIN)
""")
md("""
`eye_to_command_hops`: synapses from the eye to each command neuron (3-5). Every
photoreceptor used has a route to the commands.

## 2. Check the task (a hand-written pilot that knows where the ball is solves it; hovering or random flying does not)
""")
code("""
!python scripts/check_mac.py --env {ENV} --subset {BRAIN}
""")
md("## 3. Train the fly brain")
code("""
import re
KEEP = re.compile(r"^\\[|Eval num|Episode length|Success rate|New best")
def train(extra=""):
    cmd = (f"{sys.executable} -u scripts/train.py --env {ENV} --policy connectome --subset {BRAIN} "
           f"--profile kaggle --total_steps {TOTAL_STEPS} --seed {SEED} --device {DEVICE} "
           f"--random_start {RANDOM_START} --runs_dir {WORK}/runs --name {RUN_NAME} {extra}")
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if KEEP.search(line):
            print(line, end="", flush=True)
    if p.wait() != 0:
        print(f"!! exited with code {p.returncode}")

train()
""")
md("""
Reading the log: an `Eval` block every 50k steps (10 new rooms, real starts) and a `[progress]`
line (training episodes, including the easy starts near the ball). Hovering in place scores
about -8, random flying about -23, a solved episode about +35 to +65 (more for balls further away).
For reference, a hand-written pilot that sees only the eye image solves ~83 % (return ~40).

To continue after a timeout: raise `TOTAL_STEPS` if needed and run `train(extra="--resume")`.

## 4. Results
""")
code("""
run = f"{WORK}/runs/{RUN_NAME}"
!python scripts/plot_runs.py {run} --out {WORK}/curves_{ENV}.png
from IPython.display import Image, Video, display
display(Image(f"{WORK}/curves_{ENV}.png"))
""")
code("""
# Tests: training conditions, no wind, strong wind, 3x eye noise, 3 decoy balls,
# heavier drone, and the ball made invisible (should fail: proves it flies by sight)
!python scripts/evaluate.py --run {run} --robustness --episodes 20
""")
md("""
## 5. Videos

Top left: what the two eyes see (left eye on the left). Red = brightness (R1-6),
green = green (R8), blue = ultraviolet (R7). The yellow ball shows up yellow, the blue decoys
blue-violet, the sky pink-white.
""")
code("""
import numpy as np, imageio.v2 as imageio
os.environ["MUJOCO_GL"] = "egl"
from scripts.evaluate import load, make_norm, run_config, run_env_kw
from ctb.envs import build_env

env_name = run_config(run)["env"]
TESTS = {"clean": {}, "strong_wind": {"wind": 1.0, "gusts": 0.4}, "3_distractors": {"max_distractors": 3},
         "target_hidden": {"target_visible": False}}
model = load(run)
vn = make_norm(run, {})

def record_test(name, kw, seed=7):
    env = build_env(env_name, seed, render_mode="rgb_array", **run_env_kw(run), **kw)
    obs, _ = env.reset(seed=seed)
    frames, info = [env.render()], {}
    for t in range(env.max_steps):
        a, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
        obs, _, term, trunc, info = env.step(a[0])
        frames.append(env.render())
        if term or trunc:
            break
    frames += [frames[-1]] * 12
    path = f"{WORK}/{env_name}_{name}.mp4"
    imageio.mimsave(path, frames, fps=25)
    env.close()
    print(f"{name:14s} {t + 1}/{env.max_steps} steps, solved={info.get('is_success')}, crashed={info.get('crashed')} -> {path}")
    return path

for name, kw in TESTS.items():
    display(Video(record_test(name, kw), embed=True, width=480))
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
print(f"saved {out} ({os.path.getsize(out) / 1e6:.1f} MB); download it from the Output panel")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    src = c["source"].split("\n")
    c["source"] = [line + "\n" for line in src[:-1]] + [src[-1]]
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
json.dump(nb, open(os.path.join(root, "notebooks", "kaggle_drone.ipynb"), "w"), indent=1)
print("wrote notebooks/kaggle_drone.ipynb")
