# connectome-tray-balance

A robot arm in MuJoCo learns to keep a rolling ball on a flat tray. Its "brain" (the policy) is a recurrent network whose **wiring is copied from the fruit-fly connectome** (FlyWire v783). An MLP with the same number of parameters is the baseline.

This is a feasibility / reproduction setup, not a finished study. The goal of the first stage is simple: **the connectome policy trains and clearly beats doing nothing.**

```
robot sensors (24) ──► W_in ──► 64 input neurons ─┐
                                                  │   1000 fly neurons, ~53k real connections
                     (repeat K=4 times)  h ← (1-a)h + a·tanh(A·h + u + b)
                                                  │   A: wiring + excitatory/inhibitory sign FIXED,
64 output neurons ◄───────────────────────────────┘      only connection strengths trained
        │
        └──► linear head ──► 5 joint targets for the arm
```

## The task: `TrayBalance`

| | |
|---|---|
| Robot | 5-joint arm (shoulder yaw/pitch, elbow, wrist pitch/roll), position-controlled, tray welded to the wrist |
| Ball | 5 cm, starts near the center with a small random push |
| Observation (24) | joint angles and velocities, ball position and velocity in tray coordinates, tray normal, last action |
| Action (5) | joint target offsets around a level "home" pose |
| Reward | alive + ball near center − ball speed − jerky actions − tray drifting away; −10 if the ball falls |
| Episode | 500 steps at 50 Hz (10 s) |

A level tray is not enough: doing nothing loses the ball in about 1 second. A hand-written PD controller on the two wrist joints keeps it for the full 10 s, so the task is solvable.

## Repo layout

```
ctb/envs/tray_arm.xml        MuJoCo model (arm, tray, ball)
ctb/envs/tray_balance.py     Gymnasium environment (+ robustness knobs)
ctb/policies.py              connectome policy for Stable-Baselines3 PPO
ctb/safety.py                thread limits, duty cycle, thermal / RAM / battery guards
ctb/profiles.py              presets: mac, kaggle, smoke
scripts/download_connectome.py
scripts/build_subset.py      pick N neurons + input/output neurons
scripts/check_mac.py         one-minute health check
scripts/train.py             PPO training (connectome | mlp, real | shuffled wiring)
scripts/evaluate.py          returns, robustness tests, video
scripts/plot_runs.py         learning curves
scripts/view_env.py          live MuJoCo viewer
notebooks/kaggle_train.ipynb clone repo → download → train → evaluate
```

## On the Mac (M1, 8 GB)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_connectome.py           # ~100 MB, once
python scripts/build_subset.py --n 1000
python scripts/check_mac.py                     # ~1 min, tells you if everything works

mjpython scripts/view_env.py --mode pd          # watch the PD controller balance the ball
python scripts/train.py --policy connectome --profile smoke   # 1-2 min pipeline test
python scripts/train.py --policy connectome --profile mac     # short real run
```

### How the Mac is protected (`--profile mac`)
- only **2 threads** for PyTorch/BLAS and **2 environments**, process runs at low priority (`nice`)
- **duty cycle**: 45 s of work, then 15 s rest
- checks `pmset -g therm` every 10 s; if macOS reports throttling, it rests 2 min
- stops and saves if free RAM drops below 1 GB, or if on battery below 30 %
- stops and saves after **30 minutes** of compute (`--max_minutes` to change)
- `Ctrl+C` also saves. Continue any stopped run with `--resume`.

Everything is adjustable, e.g. `--threads 1 --work_s 30 --rest_s 30` for an even cooler run. Use the Mac to check and debug; do the long runs on Kaggle.

## On Kaggle

1. Push this repo to GitHub (e.g. `nihal4/connectome-tray-balance`).
2. Open `notebooks/kaggle_train.ipynb` in Kaggle (File → Import Notebook), turn **Internet ON**.
3. Set `REPO_URL` in the first cell, then **Save & Run All**.

The notebook trains the connectome policy and the same-size MLP, then runs robustness tests and saves models, curves and a video to `/kaggle/working`.

## Robustness tests (`scripts/evaluate.py --robustness`)

Run on policies trained only on the clean task: sensor noise (0.1, 0.3), **blind** (ball position and velocity hidden), heavy ball (3×), random pushes. These are the tests for later research questions, in the spirit of FLYNN (robustness to sensory loss).

## First test run (sanity check only)

One seed each, on a slow 2-core cloud machine, before the repo was released:

| | connectome (100k steps) | MLP (150k steps) |
|---|---|---|
| clean | 500/500 steps survived | 500/500 |
| sensor noise 0.1 | 489/500 (80 % full episodes) | 46/500 (0 %) |
| sensor noise 0.3 | 124/500 | 70/500 |
| blind | 84/500 | 79/500 |
| heavy ball / pushes | 500/500 | 500/500 |

Both policies learn the task; the connectome one needed ~50k steps. The MLP run used slightly older settings (initial action noise), so the noise gap is **not** a result yet. It needs several seeds, matched settings and the shuffled-wiring control.

## Notes and limits

- The connectome subset is the 1000 most strongly connected neurons and the connections between them. It is not a specific brain region, and the "input" and "output" neurons are chosen by graph structure, not biology.
- The network has no memory across time steps (state resets every step). Memory is a natural next experiment.
- `--wiring shuffled` keeps the same edges, signs and weights but connects them randomly; compare against it before claiming the real wiring helps.

## Data and credits

- Connectome: FlyWire, Dorkenwald et al. 2024 (*Nature*); Schlegel et al. 2024 (*Nature*).
- Processed edge list from Shiu et al. 2024 (*Nature*), github.com/philshiu/Drosophila_brain_model (MIT).
- Related work: FLYNN (Wang & Chen, arXiv 2607.00025), FlyGM (arXiv 2602.17997), Lappalainen et al. 2024 (*Nature*).
