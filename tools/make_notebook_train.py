"""Generates notebooks/kaggle_train.ipynb. Edit here, then run:
    python tools/make_notebook_train.py
"""
import os
import json
cells=[]
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n")})

md("""
# Fly-brain controller: robot arm puzzles in MuJoCo

This notebook pulls the GitHub repo, downloads the FlyWire fly-brain connectome, cuts a small sub-network out of it (the 1,000 most connected neurons), and trains it with PPO to control a MuJoCo robot arm. The policy gets the ball position and maze layout as numbers. For the camera-driven version wired through the fly's eye, use `kaggle_flyrobot.ipynb`.

**Environments** (`ENVS` below):
- `tray`: keep a ball on a flat tray (easy, solved in ~100k steps)
- `maze_fixed3`: tilt-maze puzzle, one fixed 3x3 maze (start here)
- `maze3`: a new random 3x3 maze every episode
- `maze3h` / `maze4` / `maze5`: random mazes with holes (harder each step)

**Kaggle settings (right panel):**
- **Internet: ON** (needed for git clone, pip and the connectome download)
- **Accelerator: None (CPU)** is recommended. The physics runs on CPU and the network is small, so a GPU barely helps, and this saves your GPU quota.
- Use **Save Version → Save & Run All (Commit)** for long runs, so it keeps going after you close the tab.
""")

code("""
# ===== Settings =====
REPO_URL = "https://github.com/nihal4/connectome-tray-balance.git"   # change if your repo name differs
BRANCH = "main"

ENVS = ["maze_fixed3", "maze3", "maze3h"]   # trained one after another
N_NEURONS = 1000          # size of the connectome subset (1000 = ~53k connections)
TOTAL_STEPS = 1_000_000   # per run; check the speed printed in the first run and adjust
SEED = 0
RUN_SHUFFLED = False      # same edges and weights, random wiring (control for "is it the wiring?")
PUSH_PROB = 0.0           # >0 adds random pushes on the ball (harder task)
DEVICE = "cpu"            # "cuda" only if you switched the accelerator on
""")

code("""
import os, subprocess
WORK = "/kaggle/working"
REPO = f"{WORK}/connectome-tray-balance"
if os.path.exists(REPO):
    subprocess.run(["git", "-C", REPO, "pull"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, REPO], check=True)
os.chdir(REPO)
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
""")

code("""
# torch, numpy, pandas, pyarrow, matplotlib are already on Kaggle
!pip install -q mujoco gymnasium stable-baselines3 psutil imageio imageio-ffmpeg
""")

md("## 1. Connectome: download and cut a subset")
code("""
# If you added the parquet as a Kaggle dataset, it is used instead of downloading.
import glob, shutil
os.makedirs("data/raw", exist_ok=True)
found = glob.glob("/kaggle/input/**/Connectivity_783.parquet", recursive=True)
if found and not os.path.exists("data/raw/Connectivity_783.parquet"):
    shutil.copy(found[0], "data/raw/")
    print("using", found[0])
!python scripts/download_connectome.py --out data/raw
""")
code("""
!python scripts/build_subset.py --n {N_NEURONS}
SUBSET = f"data/subset_flywire783_N{N_NEURONS}.npz"
""")

md("## 2. Health check (environment, PD controller, network speed)")
code("""
for env in ENVS:
    !python scripts/check_mac.py --subset {SUBSET} --env {env}
""")

md("## 3. Train")
code("""
import re, subprocess, sys
KEEP = re.compile(r"^\\[|Eval num|Episode length|Success rate|New best")
def train(env, policy, wiring="real", extra=""):
    cmd = (f"{sys.executable} -u scripts/train.py --env {env} --policy {policy} --wiring {wiring} "
           f"--profile kaggle --subset {SUBSET} --total_steps {TOTAL_STEPS} --seed {SEED} "
           f"--push_prob {PUSH_PROB} --device {DEVICE} --runs_dir {WORK}/runs {extra}")
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:                      # show only the useful lines, live
        if KEEP.search(line):
            print(line, end="", flush=True)
    if p.wait() != 0:
        print(f"!! {env}/{policy} exited with code {p.returncode}")

for env in ENVS:
    train(env, "connectome")
    if RUN_SHUFFLED:
        train(env, "connectome", "shuffled")
""")

md("""
If a run stops early (time limit, session restart), call e.g. `train("maze3", "connectome", extra="--resume")`; it continues from the last checkpoint in `/kaggle/working/runs`.

## 4. Results
""")
code("""
runs = sorted(glob.glob(f"{WORK}/runs/*"))
from IPython.display import Image, display
for env in sorted({os.path.basename(r).split("-")[0] for r in runs}):
    group = " ".join(r for r in runs if os.path.basename(r).startswith(env + "-"))
    !python scripts/plot_runs.py {group} --out {WORK}/curves_{env}.png
    display(Image(f"{WORK}/curves_{env}.png"))
""")
code("""
# Robustness tests the policies never saw in training: sensor noise, hidden ball ("blind"), heavy ball, pushes
for r in runs:
    print("\\n#####", os.path.basename(r))
    !python scripts/evaluate.py --run {r} --robustness --episodes 10
""")
code("""
# One video per run (needs offscreen OpenGL; skipped if unavailable)
os.environ["MUJOCO_GL"] = "egl"
from IPython.display import Video
for r in runs:
    vid = f"{WORK}/{os.path.basename(r)}.mp4"
    !python scripts/evaluate.py --run {r} --episodes 1 --video {vid}
    if os.path.exists(vid):
        print(os.path.basename(r)); display(Video(vid, embed=True, width=420))
""")
code("""
# Export everything (runs, videos, curves) as one zip
import zipfile
out = f"{WORK}/runs_export.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{WORK}/runs/**/*", recursive=True):
        if os.path.isfile(f):
            z.write(f, os.path.relpath(f, WORK))
    for f in glob.glob(f"{WORK}/*.mp4") + glob.glob(f"{WORK}/*.png"):
        z.write(f, os.path.join("media", os.path.basename(f)))
print("saved", out, f"({os.path.getsize(out) / 1e6:.1f} MB) - download it from the Output panel")
""")

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
for c in nb["cells"]:
    src=c["source"].split("\n"); c["source"]=[l+"\n" for l in src[:-1]]+[src[-1]]
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
json.dump(nb, open(os.path.join(root, "notebooks", "kaggle_train.ipynb"), "w"), indent=1)
print("wrote notebooks/kaggle_train.ipynb")
