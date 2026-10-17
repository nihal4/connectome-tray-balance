# connectome-tray-balance

**Fly-brain-wired controllers for robot-arm balance and maze puzzles**

A robot arm in MuJoCo learns to balance a ball on a tray and then to solve **tilt-maze puzzles** (roll the ball through a maze to the goal). Its "brain" (the policy) is a recurrent network whose **wiring is copied from the fruit-fly connectome** (FlyWire v783). Only the fly-brain network is studied here.

This is a feasibility / reproduction setup, not a finished study.

**Project page with videos:** https://nihal4.github.io/connectome-tray-balance/ (from `index.html`; enable it under Settings → Pages → Deploy from branch `main`, folder `/ (root)`).

![maze](results/previews/maze4.png)

```
robot sensors (24-66) ─► W_in ─► 64 input neurons ─┐
                                                  │   1000 fly neurons, ~53k real connections
                     (repeat K=4 times)  h ← (1-a)h + a·tanh(A·h + u + b)
                                                  │   A: wiring + excitatory/inhibitory sign FIXED,
64 output neurons ◄───────────────────────────────┘      only connection strengths trained
        │
        └──► linear head ──► 5 joint targets for the arm
```

## Fly-robot mode: camera into the eye, commands out of the command neurons

The second way to use the connectome. Instead of the 1000 most connected neurons with
inputs and outputs picked by us, the robot plugs into the fly's own channels:

| robot | fly brain (FlyWire cell types) | count |
|---|---|---|
| 16x16 wrist camera, this frame | R8 photoreceptors, right eye | 256 |
| same camera, previous frame (motion) | R7 photoreceptors, right eye | 256 |
| joint angles/speeds, tray tilt | ascending neurons (body sense from the nerve cord) | 64 |
| arm commands | descending neurons (the brain's commands to the body) | 64 |
| in between | neurons on the shortest pathways: optic lobe, visual projection, central brain | ~1,900 |

Each pixel drives exactly one photoreceptor (no mixing layer), arranged roughly like the
eye. Every eye-to-command path inside the network is 3-5 synapses long. The brain is not
given the ball position, the wall map or the goal: it only sees the camera image and feels
its own joints.

```bash
python scripts/download_connectome.py          # also gets the FlyWire cell-type annotations
python scripts/build_flybrain.py               # -> data/flybrain_cam16_N2500.npz
python scripts/train.py --env tray_vision --policy connectome --subset data/flybrain_cam16_N2500.npz --profile kaggle
```

Kaggle: `notebooks/kaggle_flyrobot.ipynb`. Videos show the camera
image the brain receives in the top-left corner. Camera tasks: `tray_vision`,
`maze_fixed3_vision`, `maze3_vision`, `maze3h_vision`; a controller that uses only the camera
image solves all of them 100 %, so the picture contains everything needed.

Settings that differ automatically for this brain: 8 recurrent rounds (signals need 3-5 hops),
no leak, stronger initial connections (otherwise the camera signal fades before reaching the
command neurons), and a sparse matrix (0.5 % of neuron pairs are connected).

## Fly-brain drone: compound eye, fly wiring, flight commands

The third robot (`ctb/envs/drone.py`, notebook `notebooks/kaggle_drone.ipynb`). A fly's brain
evolved to steer flight, so here its command neurons steer a small quadcopter.

| drone | fly brain (FlyWire cell types) | count |
|---|---|---|
| compound eye, brightness per facet | R1-6 photoreceptors, both eyes | 2 x 160 |
| compound eye, ultraviolet per facet | R7 photoreceptors | 2 x 160 |
| compound eye, green per facet | R8 photoreceptors | 2 x 160 |
| airspeed, tilt, rotation, last command | ascending neurons | 64 |
| flight commands: forward, sideways, up, turn | descending neurons (32 per side) | 64 |
| in between | pathway neurons: optic lobe, visual projection (incl. LC/LPLC), central brain | ~2,900 |

**Eye.** Two eyes of 160 facets each cover almost all directions (from 15 deg across the
midline to 165 deg behind, 75 deg below to 80 deg above the horizon, about 11 deg between facets,
as in a fly the image is very wide and very blurry). The room is ray traced (7 rays per facet,
0.5 ms per step, no graphics card needed); balls are blended into every facet through the
facet's blur profile, so a far ball is a faint smudge and grows as the drone gets closer. Each
photoreceptor gets the light it sees and how much that just changed (real photoreceptors
respond strongly to change; this is what motion and looming detection build on).
Facets are matched to photoreceptors by position on the eye (approximate retinotopy from neuron
coordinates, optimal assignment), only photoreceptors that have a route to the chosen command
neurons are used, and the pathway growth keeps a route for every facet: all 960 photoreceptors
and all 64 body-sense inputs reach the commands in 3-5 synapses.

**Stabiliser = halteres.** A fly's balance reflexes run outside the brain (halteres, nerve
cord), and FlyWire maps the brain only. So a simple controller keeps the drone level and holds
the velocity and turn rate the brain asks for; the brain decides where to fly.

**Speeds, chosen to keep flying smooth:** max 1 m/s horizontal, 0.5 m/s vertical, 86 deg/s
turning; commanded velocity changes by at most 2 m/s^2 (0 to full speed in 0.5 s), tilt at most
25 deg. Brain runs at 25 Hz, stabiliser at 500 Hz.

**Task `drone_reach`.** 8 x 8 m grey room (low-contrast checker floor, striped walls, so the
eye can sense its own motion), open sky. A yellow ball (25 cm radius) appears 1.5-4 m away, anywhere in the room; the drone starts at a random place
and heading. Solved: drone centre within 30 cm of the ball, slower than 0.5 m/s, for 0.4 s.
12 s limit.

| reward term | value | why |
|---|---|---|
| getting closer | +10 per m (potential) | flying back and forth earns nothing |
| hovering on the ball | +0.5 per step (potential) | leaving gives it back, so no farming |
| solved | +20 | |
| time | -0.01 per step | |
| crash (floor, wall, above 3 m, flipped) | -10, episode ends | crashing is always worse than hovering |

No penalty on jerky commands: the stabiliser already limits acceleration, and in a first test such
a penalty mostly punished PPO's exploration noise (about -13 per episode), drowning out the
"get closer" signal; training return stayed at random-flying level for 100k steps.

**Colours (what each photoreceptor type sees).** The yellow ball is bright in R1-6 and R8 and
almost black in R7 (UV); the grey room reflects UV and green alike; the sky is UV-rich; decoys are
UV-violet. In a first version the floor was green and the ball yellow-green: the two had nearly
the same colour ratio, so a far ball against the floor was camouflaged. A detector on the eye
image picked the ball only 44 % of the time at 3-4 m; with the grey room 76 % from a single look.

**Built-in robustness (training conditions):** a new layout every episode, light wind and gusts
in random directions, drone mass +-4 %, eye noise (0.01 per photoreceptor), light level +-20 %,
floor pattern size, and up to two blue-violet decoy balls (as bright as the target, other colour).
Tests (`evaluate.py --robustness`): no wind, strong wind, 3x eye noise, three decoys, heavier
drone, and the ball made invisible (must fail).

**Checks before training** (60 rooms each):

| pilot | solved | crashed | return |
|---|---|---|---|
| hand-written, knows where the ball is | 100 % | 0 % | 52 |
| hand-written, sees ONLY the eye image (`EyeOnlyPilot`) | 83 % | 10 % | 40 |
| same, strong wind | 68 % | 13 % | 33 |
| same, ball invisible | 0 % | 25 % | -8 |
| hover in place | 0 % | 22 % | -8 |
| random flying | 0 % | 27 % | -23 |

The eye-only pilot shows the compound eye carries enough information; the invisible-ball row
shows the task cannot be solved without seeing the ball.

**Local learning check** (2-core machine, same settings as the notebook): training return rose
from -14 to about -3 and training episodes solved from 3 % to 11 % by 150k steps; the first
evaluation room (real start, unseen layout) was solved at 200k steps. Early, but learning.

```bash
python scripts/build_flybrain.py --eye compound --facets 160 --n 4000 --hops 5   # -> data/flybrain_eye160_N4000.npz
python scripts/check_mac.py --env drone_reach --subset data/flybrain_eye160_N4000.npz
python scripts/train.py --env drone_reach --subset data/flybrain_eye160_N4000.npz --profile kaggle --random_start 0.5
```

## Fly-brain Panda: a real robot arm with a gripper (`panda_reach`)

The current main robot (`ctb/envs/panda.py`). A Franka Emika Panda arm with its two-finger
gripper (model from MuJoCo Menagerie, Apache-2.0; `python scripts/download_panda.py`), set up
the way the standard manipulation benchmarks do it (Fetch, robosuite, ManiSkill):

| | design |
|---|---|
| brain commands (descending neurons) | `[dx, dy, dz, grip]` 25 times a second: move the gripper up to 2 cm, open/close. Inverse kinematics + the arm's servos do the joints, like a nerve cord turning descending commands into leg movements |
| body sense (ascending neurons) | gripper position and velocity, gripper opening, last command |
| right eye | compound eye (160 facets) on a "head" beside the table, looking at the workspace: finds the block |
| left eye | compound eye (160 facets) on the wrist, looking past the fingers: lines the gripper up |
| brain | the same 4,000-neuron fly-brain network as the drone (`flybrain_eye160_N4000.npz`) |
| task `reach` | bring the gripper to the yellow 5 cm block (new place every episode, up to 2 blue decoys) and hold still in the pre-grasp pose, 1.5 cm above it |
| episode | 60 steps (2.4 s), no early end |
| reward (ManiSkill style, per step, max 1) | reach `1 - tanh(5 d)`; within 3 cm also `1 - tanh(10 speed)`; success (within 3 cm, slower than 5 cm/s) = maximum. Bounded, nothing to farm |

Why two eyes: a fly-like eye has ~11 deg facets; seen from beside the table, a 5 cm block is a
faint dot inside one facet, and an eye-only detector placed it with a median error of ~5 cm
(the task needs 3 cm). Robot labs pair an overview camera with a wrist camera for the same
reason. With the left eye on the wrist, the block fills many facets up close.

Checks (40 episodes each):

| pilot | solved |
|---|---|
| hand-written, knows where the block is | 100 % |
| hand-written, sees ONLY the two eye images (+ own hand position) | 100 % (also with decoys) |
| same, block invisible | 0 % |
| do nothing / random | 0 % / 0 % |

Two problems found and fixed while building it: the hand lagged behind its commands (fixed
with a target point the hand follows, as in Fetch), and in a first version the eye sat behind
the robot, where the arm blocked its view of the table.

Planned ladder: reach (pre-grasp) -> push a block onto a goal -> colour sorting by pushing ->
pick and place with the gripper.

**How others connect fly brains to bodies** (checked before building):
- NeuroMechFly v2 (Wang-Chen et al. 2024, *Nature Methods*): the brain-level controller sends a
  two-number descending signal into leg pattern generators; the eye model bins a camera image
  into ~721 ommatidia. Layered control, as here.
- FlyGM (arXiv 2602.17997): whole-brain FlyWire graph with fixed signed synapse counts; observations
  enter through afferent neurons, actions are read from efferent neurons; neuron state carries
  over time (memory); imitation of an expert first, then PPO.
- FLYNN (arXiv 2607.00025): connectome-derived network for vision-based navigation, robust to
  vision loss.
- FlyCNS (arXiv 2609.28816): connectome statistics used only as a prior on brain-body
  communication.
- No published fly-brain controller for a robot arm was found.

## Environments (`--env`)

| name | task | obs | time limit | holes |
|---|---|---|---|---|
| `tray` | keep a rolling ball on a flat tray | 24 | 500 steps (10 s) | – |
| `maze_fixed3` | tilt-maze puzzle, one fixed 3x3 maze | 47 | 300 | 0 |
| `maze3` | a **new random 3x3 maze every episode** | 47 | 350 | 0 |
| `maze3h` | random 3x3 mazes with holes | 47 | 350 | 2 |
| `maze4` | random 4x4 mazes | 66 | 550 | 3 |
| `maze5` | random 5x5 mazes with some loops | 91 | 750 | 5 |

Shared by all: a 5-joint arm (shoulder yaw/pitch, elbow, wrist pitch/roll), position-controlled, holding a tray on a short post at 50 Hz. The action (5) is joint-target offsets around a level home pose. The observation always starts with joint angles and velocities, the ball's position and velocity in tray coordinates, the tray normal, and the last action.

**Tray balance.** The tray has no rim. Doing nothing loses the ball in about 1 s; a hand-written PD controller keeps it for the full 10 s.

**Tilt maze (the puzzle).** Like the wooden labyrinth toy: walls, a glass lid, holes, and a green goal. The maze is generated by depth-first search, so there is exactly one route, the ball starts in the cell farthest from the goal, and from `maze3h` on, holes sit in cells beside the route, so a wrong turn or an overshoot drops the ball. **Solved** means the ball rests in the goal circle for 0.5 s; rolling through it does not count. The policy sees one bit per possible inner wall, one bit per hole and the goal position, but **not** the route: it has to work out the detours from the layout. Reward is progress along the shortest route (continuous across cells), an extra pull toward the goal centre inside the goal cell, progress toward the 0.5 s hold in the goal (taken back if the ball leaves, so hovering earns nothing), +10 when solved, −10 for a hole, and small costs for time, jerky actions and tray drift.

Reference controllers, 20 episodes each (`python scripts/check_mac.py --env maze4` prints the first two):

| | scripted planner (knows the route) | do nothing | random actions | random ±1 shaking |
|---|---|---|---|---|
| `maze_fixed3` | 100 % | 0 % | 0 % | 0 % |
| `maze3` | 100 % | 0 % | 0 % | 0 % |
| `maze3h` | 100 % | 0 % | 0 % | 0 % |
| `maze4` | 100 % | 0 % | 0 % | 0 % |
| `maze5` | 100 % | 0 % | 0 % | 0 % |

Two shortcuts were closed while building this: without the lid, a policy learned to flick the ball over the walls; without holes, the hold rule and tight time limits, random shaking solved 40–70 % of small mazes by luck.

Start-position curriculum: `--random_start 0.5` makes half of the training episodes start the ball at a random cell along the route, so the policy also practises the final turns and stopping in the goal (evaluation always uses the real start). Added after the camera maze run reached the goal but pinned the ball in the goal-cell corner instead of stopping it in the centre.

Slow, smooth motors: `--action_rate 0.1` limits how much each joint target may change per step (a full swing takes 0.4 s, like a servo with a speed limit). Under hard shaking the ball's top speed drops from 2.35 to 0.71 m/s; the scripted planner still solves every maze. It changes the robot, so it is saved in the run's `config.json` and evaluation and videos use it automatically; keep the same value when resuming.

Suggested curriculum: `maze_fixed3` (one puzzle) → `maze3` (generalise to new puzzles) → `maze3h` (holes) → `maze4` → `maze5`. Holes start later on purpose: in a first test, PPO with holes from the start spent its first 200k steps mostly learning to avoid them. Robustness tests apply to every environment; for `maze_fixed3` runs there is also an **unseen mazes** test.

## Repo layout

```
ctb/envs/tray_arm.xml        MuJoCo model (arm, tray, ball)
ctb/envs/tray_balance.py     TrayBalance env (+ robustness knobs, PD controller)
ctb/envs/labyrinth.py        tilt-maze puzzle env (maze generator, scripted planner)
ctb/envs/drone.py            fly-brain drone: quadcopter, stabiliser, compound eye, reach task
ctb/envs/panda.py            fly-brain Panda: arm + gripper, hand-space control, head + wrist compound eyes
ctb/envs/__init__.py         registry: build_env("maze3"), make_env(...)
ctb/policies.py              connectome policy for Stable-Baselines3 PPO
ctb/safety.py                thread limits, duty cycle, thermal / RAM / battery guards
ctb/profiles.py              presets: mac, kaggle, smoke
scripts/download_connectome.py
scripts/download_panda.py    Panda model from MuJoCo Menagerie
scripts/build_subset.py      pick N neurons + input/output neurons
scripts/check_mac.py         one-minute health check
scripts/train.py             PPO training of the fly-brain network (real | shuffled wiring)
scripts/evaluate.py          returns, robustness tests, video
scripts/plot_runs.py         learning curves
scripts/view_env.py          live MuJoCo viewer
scripts/desktop/             always-on Ubuntu PC: setup.sh, fly command, auto-resume service, progress videos + dashboard
scripts/watch_maze.py        maze diagnostic videos (real start, and k cells before the goal)
scripts/build_flybrain.py    fly-robot brain: eye/body-sense inputs, command outputs, pathways
                             (--eye compound: the drone's two compound eyes)
notebooks/kaggle_train.ipynb 1,000-neuron network, ball/maze given as numbers
notebooks/kaggle_flyrobot.ipynb  fly-robot brain, camera tasks (train, test videos, export zip)
notebooks/kaggle_drone.ipynb fly-brain drone with compound eyes
tools/make_notebook_*.py     generate the notebooks (edit there, not in the .ipynb)
results/                     saved results (models, curves, videos) per experiment
```

## On the Mac (M1, 8 GB)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_connectome.py           # ~100 MB, once
python scripts/build_subset.py --n 1000
python scripts/check_mac.py --env maze3         # ~1 min, tells you if everything works

mjpython scripts/view_env.py --env maze4 --mode scripted     # watch the planner solve a maze
python scripts/train.py --env maze_fixed3 --policy connectome --profile smoke   # 1-2 min pipeline test
python scripts/train.py --env maze_fixed3 --policy connectome --profile mac     # short real run
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
3. Set `REPO_URL` and `ENVS` in the first cell, then **Save & Run All**.

For each environment in `ENVS` the notebook trains the fly-brain policy (optionally the shuffled-wiring control), then runs robustness tests and saves models, curves and one video per run to `/kaggle/working`. Download `runs_export.zip` and put interesting runs under `results/`.

For the fly-robot (camera) tasks use `notebooks/kaggle_flyrobot.ipynb` instead. Its settings cell picks the task (`ENV`), brain size, `ACTION_RATE` (slow motors, default 0.1) and `RANDOM_START` (curriculum, default 0.5). Runs with slow motors are named `<env>-connectome-real-s0-slow`, so they never mix with the older fast-motor runs. The notebook trains, tests, records one video per test plus maze diagnostics, and zips everything.

## On an always-on Ubuntu PC (`scripts/desktop/`)

For runs of days or weeks (tested target: Ubuntu 24.04, i3-8100, 12 GB). One-time setup:

```bash
git clone https://github.com/nihal4/connectome-tray-balance ~/flybrain/repo
bash ~/flybrain/repo/scripts/desktop/setup.sh
```

It installs CPU PyTorch + MuJoCo in `~/flybrain/.venv`, downloads the connectome, builds the
brains, runs the check, and sets up three services: training (started only with `fly start`,
never at boot), (`--profile desktop`: full speed, no time limit, cools down above 90 C, stops on low
RAM), plus two that start at boot: a watcher that records a progress video of the newest checkpoint every 100k steps, and a
dashboard at `http://<pc>:8080` (curves, latest numbers, CPU temperature, videos).

Control it over SSH with `fly`: `fly job maze3_vision 3000000`, `fly start`, `fly status`,
`fly log`, `fly eval`, `fly stop`, `fly update`.

Power cuts: checkpoints every 50k steps are written to a temporary file and swapped in, with
the previous one kept as `.bak` (used automatically if the newest is unreadable). After a reboot
or power cut, `fly start` resumes from the last checkpoint (training never starts by itself).

## Robustness tests (`scripts/evaluate.py --robustness`)

Run on policies trained only on the clean task: sensor noise (0.1, 0.3), **blind** (ball position and velocity hidden), heavy ball (3×), random pushes. These are the tests for later research questions, in the spirit of FLYNN (robustness to sensory loss).

## Results so far

- **v1, tray balance** (`results/v1_tray_balance/`): the 1,000-neuron network solves it by 100k steps and is robust to noise, a heavy ball and pushes; it fails when the ball is hidden.
- **v2, fixed maze** (`results/v2_maze_fixed3_local/`, local, 400k steps): not solved yet; it often reaches the goal but cannot stop the ball there.
- **v3, Kaggle mazes** (`results/v3_maze_kaggle/`, 1M steps): fixed maze 30–40 % solved by 1M and rising; random mazes 0 %. Also exposed a reward hack (hovering in the goal), fixed afterwards.
- **v4, fly-robot brain by vision** (`results/v4_flyrobot_tray_vision_local/`, local, one seed): wired from the eye to the command neurons and seeing only a 16x16 camera image, the fly brain learned to keep the ball on the tray at ~200k steps; best model 100 % on new episodes, 0 % when the ball is hidden from the camera.
- **v5, fly-robot brain by vision, Kaggle** (`results/v5_flyrobot_tray_vision_kaggle/`, 1M steps, ~2.7 h): full 10 s from 250k steps on; 100 % clean, with a 3x heavier ball and with random pushes; 0 % with camera noise or the ball hidden. Videos of all three tests included.
- **v6, camera maze, first run** (`results/v6_maze_vision_kaggle/`, Kaggle, one seed): 0 % solved, return flat at about +5.3 from 1.05M steps. The video shows why: the brain rolls the ball through the whole maze to the goal in under a second, then pins it in the corner of the goal cell instead of stopping it in the middle. Navigation learned, final stop not. Fixes: start-position curriculum and a stronger pull to the goal centre (success rule unchanged).
  Resumed with both fixes from 1.65M to ~2M steps: eval return rose from ~5 to 7.7–11.2 (the ball now comes off the wall towards the centre) but still 0 % solved; the arm and ball move too fast to stop the ball on the spot.
- **v7, camera maze solved with slow motors** (`results/v7_maze_vision_slow_kaggle/`, Kaggle, 2M steps, one seed): fresh run with `--action_rate 0.1 --random_start 0.5`, brain unchanged. First solve at 350k, 100 % from 1.1M, 100 % on every eval from 1.65M; solves in ~130 steps and stops the ball in the goal centre. Robustness: heavy ball 100 %, pushes 60 %, camera noise 0 %, ball hidden 0 %, **unseen mazes 0 %** (memorised this maze).
- **Next:** `maze3_vision` (new random maze every episode) to see whether the brain can learn to read walls through the eye.
- **Drone:** `drone_reach` with compound eyes (see above), ready to train with `notebooks/kaggle_drone.ipynb`.

## Notes and limits

- The arm's forearm visibly passes through the tray in some poses. It is cosmetic: arm parts are in a separate collision group and never touch the ball.
- The connectome subset is the 1000 most strongly connected neurons and the connections between them. It is not a specific brain region, and the "input" and "output" neurons are chosen by graph structure, not biology.
- The network has no memory across time steps (state resets every step). That is enough for the mazes because the layout is in every observation; memory is a natural next experiment (e.g. show the layout only at the start).
- `--wiring shuffled` keeps the same edges, signs and weights but connects them randomly; compare against it before claiming the real wiring helps.

## Data and credits

- Connectome: FlyWire, Dorkenwald et al. 2024 (*Nature*); Schlegel et al. 2024 (*Nature*).
- Processed edge list from Shiu et al. 2024 (*Nature*), github.com/philshiu/Drosophila_brain_model (MIT).
- Related work: FLYNN (Wang & Chen, arXiv 2607.00025), FlyGM (arXiv 2602.17997), Lappalainen et al. 2024 (*Nature*).
