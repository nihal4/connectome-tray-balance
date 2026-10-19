# v9: Panda arm grasps and lifts a block it finds with fly compound eyes (desktop, one seed)

Run: `panda_lift-connectome-real-s0-ft`, 3.4M steps, pure reinforcement learning (PPO), no
imitation. Started from the v8 reaching brain (`--init_from`, same 4,000-neuron FlyWire brain,
same eyes and body sense, plus touch). Task `panda_lift`: grasp the yellow 5 cm block (new place
every episode, 0-2 blue decoys) between both fingers, lift it 8 cm and hold it there; solved =
lifted and held at the END of the 100-step (4 s) episode. Reward: height-based staged reward
(reach, +1 held, +2 x lift progress while held).

## What it took (in order)

| steps | change | effect |
|---|---|---|
| 0-0.7M | start from the reach brain, lr 3e-4 | goes down to the block (return 14 -> 21), never grasps |
| 0.75M | 30 % of practice episodes start mid-task (at the block, open or already holding) | learns to close and hold (training return 22 -> 27.5) |
| 1.1M | exploration noise back to 0.3 once, lr 1e-4, 50 % mid-task starts | aim restored (it had drifted to decoys); grasps in normal starts; lifts ~2.5 cm |
| 2.15M | (no change) | **first solved evaluation episodes** |
| 2.4M | | 12 % solved |
| 3.2M | | best brain: eval return 35.6, 26 % solved |
| 3.3M | | 38 % solved in one evaluation (best single reading) |

Evaluation solved rate, mean of the 13 evaluations from 2.8M to 3.4M: **22.8 %** (50 episodes
each). Training return rose almost linearly from 14 to ~52; it levelled off around 3.1M while the
exploration noise kept shrinking (0.30 -> 0.15), so the run was stopped at 3.4M.

## Robustness (best brain, 50 episodes each, `fly eval 50`)

| Test | Solved | Return |
|---|---|---|
| clean | 16 % | 34.2 |
| no decoys | 24 % | 37.3 |
| brightness +-40 % (training +-20 %) | 20 % | 30.5 |
| eye noise x3 | 8 % | 19.1 |
| yellow block invisible | 0 % | 6.1 |

It steers by its eyes (0 % with the block hidden). Decoys cost a little; heavy eye noise more.

## What the successes look like (`showcase_best_brain.mp4`)

`fly showcase` runs the best brain on new random layouts without a camera, then replays the
solved ones with the camera (the brain is deterministic, so the replays are identical episodes).
On 40 new layouts it solved 5 (12 %) at 2.5M. In the clips: it finds the yellow block (also next
to blue decoys), lines up, closes both fingers, lifts and holds. One lift goes almost straight
up; others swing the block sideways while lifting - the height-only reward does not penalise
that. The goal reward (`--lift_reward goal`, block to a point straight above its start) was
built for exactly this.

## Honest notes

* One seed; 50-episode evaluations swing by about +-10 points (16 % and 38 % both occurred).
* Practice episodes included easier starts (30 % near the block, 50 % mid-task); evaluation and
  videos always start from scratch.
* Weak spot carried over from reaching: the far-right corner of the table.

Files: `showcase_best_brain.mp4` (3 solved + 1 missed, best brain), `showcase_2500k.mp4`,
`progress_3400k_fixed_layouts.mp4` (the 3 fixed test layouts), `curves.png`, `eval_results.json`.
