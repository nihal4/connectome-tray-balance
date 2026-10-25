# v18: greenhouse phase 2b, pick every ripe tomato (two cameras, memory, gentle; desktop, one seed)

Task: a truss of three cherry tomatoes (6.5 cm apart); one or two are ripe, the others green or turning. Pick
every ripe one into the basket, one at a time, and leave the rest. Solved = all ripe tomatoes in the basket,
nothing unripe picked, and **no tomato bruised** (gentle handling, below).

Final brain: `panda_pickripe-connectome-real-s0-2cam-2ripe-mem-zoom` (best brain). Three runs led to it:

```
# 1. two cameras + memory + gentle, from the v17 brain: 10 teacher rounds, then PPO (2M steps)
fly job panda_pickripe 2000000 "--init_from panda_pickripe-connectome-real-s0-cam-1ripe --imitate 10 --privileged
  --grip_mode absolute --eye_adapt --wrist_cam rgbd --head_cam rgbd --ripe_max 2 --return_start 0.3
  --memory --gentle --lr 1e-4"
# 2. smooth fine-tune (touch penalty, episode ends when done)
fly job panda_pickripe 1000000 "--init_from panda_pickripe-connectome-real-s0-2cam-2ripe-mem --smooth --lr 5e-5
  --return_start 0.3 --eval_episodes 20"
# 3. zoomed front camera (26 deg instead of 34)
fly job panda_pickripe 1000000 "--init_from panda_pickripe-connectome-real-s0-2cam-2ripe-mem-smooth
  --head_fovy 26 --lr 5e-5 --return_start 0.3 --eval_episodes 20"
```

## What the robot has

- **Wrist depth camera** (as in v17): sees the pick and the drop.
- **Front depth camera** (`--head_cam rgbd`, new): fixed on the cart beside the arm, facing the truss, like the
  RealSense D435 harvesting robots use to find fruit. Its 16 x 10 image (red, nearness, green) replaces the
  head fly eye in the same 480 input neurons, mapped by layout.
- **Smell** (106 receptor neurons, unchanged) and **memory** (`--memory`): each neuron keeps a learned share of
  its activity from one step to the next (starts at 5 %), trained with recurrent PPO and sequence imitation.
- **Gentle handling** (`--gentle`): gripper force capped at 3 N (median squeeze 1.5 N, below a tomato's
  ~2.6 N bio-yield), release 7 cm above the basket, the teacher slows near the fruit; any robot contact on a
  tomato above 5 N, or a landing faster than 1 m/s, bruises it and fails the episode.
- **Observation pose**: after each tomato in the basket, the teacher goes back to a pose that puts the wrist
  camera 25 cm above the truss area, looking at it, rising out of the basket first (no knocks on the way).

## Result (`scripts/precision_pickripe.py`, 200 new layouts, the same for every brain)

| Brain | Solved | 1 ripe | 2 ripe | Unripe picked | Bruised | Touches others | Time |
|---|---|---|---|---|---|---|---|
| 2 cameras + memory, best (1.37M) | **82 %** | 89 % | 76 % | 2 % of layouts | 11 % | 34 % (1.5 steps) | 131 steps |
| same, latest (2M) | 77 % | 82 % | 73 % | 1 % | 10 % | 24 % (0.6) | 124 |
| + smooth (best = 50k of it) | 79 % | 87 % | 73 % | 4 % | 9 % | 28 % (2.6) | 123 |
| **+ zoomed front camera (final)** | **79 %** | 86 % | 74 % | **0.5 %** (1 of 290) | 10 % | 25 % (0.8) | **121.5 (4.9 s)** |

For comparison, phase 2b without memory and front camera reached ~66 % (dashboard, 10-layout tests).
All differences between the four brains in "solved" are within the ±4-point noise of a 200-layout test; the
zoom clearly fixed the red/orange mix-up (unripe picks 6-8 -> 1).

Senses (final brain, 50 layouts each): antennae silenced 0 % (smell times the grasp); every tomato looking ripe
30 % with unripe picked in 58 % of layouts (colour decides which); turning tomatoes made to smell ripe 78 %,
unripe in 4 % (it double-checks with colour). Earlier `fly eval` of the 2-camera memory brain: memory wiped
every step 62 % vs 74 % (it uses its memory); tinted / uneven light 0 % (no white balance yet).

## What went wrong on the way, and how each was found

1. **Memory diverged** (loss -> NaN, arm crash): full carry-over scrambled the copied brain's dynamics. Fix: a
   learned per-neuron carry-over gate starting at 5 % (like an LSTM forget gate), NaN guards.
2. **Second tomato never taken**: reaching it meant remembering its place over ~3 s of carrying, longer than the
   48-step training pieces teach. Fix: the observation pose (how real harvesters work).
3. **Return trip knocked tomatoes** (3 % of episodes, up to 80 N). Fix: rise out of the basket first.
4. **From the observation pose the wrist camera saw the next tomato in 35 % of layouts** (it looks almost
   straight down, not along the fingers). Fix: a pose aimed by the camera's own line of sight (98 %), and the
   front camera.
5. **PPO slowly unlearned** (90 % -> 60 % at lr 3e-4). Fix: lr 1e-4, then 5e-5.
6. **Physics blow-ups** ("Nan, Inf or huge value"): a knocked tomato flew, sank into the table, then MuJoCo
   blew up. Fix: a tomato faster than 2.5 m/s or below the table ends the episode as lost.
7. **Training too slow** (12 steps/s): recurrent PPO replayed each episode from its first step; now 32-step
   pieces from stored neuron states (RLlib / R2D2 style), exact to 5e-5: 6-7x faster updates.
8. **Heading for the orange tomato first** (the 500k smooth video): at 34 deg a tomato is ~1.4 front-camera
   pixels; red vs orange at the tomato pixel 93 %. Zoom to 26 deg: 99 %, every tomato still in view.

## Where it still fails (final brain, `scripts/failures_pickripe.py`, same 200 layouts)

| Cause | Layouts |
|---|---|
| bruised a tomato | 19 |
| ripe one picked but dropped outside the basket | 8 |
| second ripe one never picked | 8 |
| first ripe one never picked | 5 |
| tomato flung | 1 |
| unripe picked | 1 |

First bruises: approaching the ripe one it is picking (9, mostly the left finger), closing on it (6), knocking a
neighbour on the way in (5), while carrying (4). A guarded move (stop pushing into a tomato pressed above 2 N)
changed little (bruise failures 11 -> 11 on 80 layouts): the bruises are impacts that peak within one 40 ms
control step. The final approach needs to be more precise or slower; a limit of this phase.

## Files

- `progress_1000k_zoom.mp4`: the final run's test layouts at 1M steps (all three solved, the two-ripe one 2/2)
- `showcase_2cam_mem.mp4`: the 2-camera memory brain on new layouts (before smooth and zoom)
- `precision_summary.json`: every number above; `eval_results_2cam_mem.json`: `fly eval 50` of run 1
- `curves.png`: the zoom fine-tune; `config_zoom.json`
