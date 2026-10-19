# v10: Panda grasp and lift, teacher first, then practice (desktop, one seed)

Run: `panda_lift-connectome-real-s0-imitate`, stopped at 2.12M steps (the score had levelled off).
Same 4,000-neuron FlyWire brain, same compound eyes, body sense and touch as v9, started from the
v9 brain.

```
fly job panda_lift 4000000 "--init_from panda_lift-v9 --imitate 10 --privileged --grip_mode absolute
  --lift_reward goal --edge_bias 0.3 --random_start 0.3 --std 0.15 --lr 1e-4 --eval_episodes 50"
```

Task: grasp the yellow block (new place every episode, 0-2 blue decoys), bring it to the point
8 cm straight above where it lay and hold it there (within 2.5 cm for 0.2 s); solved = held there at
the end of the 100-step (4 s) episode.

## How it was trained

1. **Teacher rounds (DAgger, 165k steps).** A scripted teacher that knows the true block position
   labels every step with its own move; who drives goes from the teacher (100 %) to the fly brain
   (50 %, 25 % ...). The fly brain learns to output the teacher's moves from its eyes and body sense.
2. **Practice (PPO, ~1.95M steps).** Trial and error from there, with a critic that sees the true
   state during training (asymmetric actor-critic). The fly brain itself never receives the block
   position; the teacher and the critic are not used in tests, maps or videos.

| Round | Teacher drives | Student alone: solved |
|---|---|---|
| 1-2 | 100 %, 50 % | 0 % |
| 3 | 25 % | 18 % |
| 5 | 6 % | 28 % |
| 7 | 2 % | **56 %** |
| 9-10 | ~0 % | 2-4 % |

After round 7 the data filled with the student's own near-misses (capped at 120k examples), the
fit got worse (imitation loss 0.033 -> 0.078) and it lost the final lift while keeping a clean reach
and grasp (`progress_0100k_after_imitation.mp4`). Practice started from that round-10 brain and
fixed it: 54 % at 265k, 84 % at 565k, 84-96 % on every evaluation from 715k on (mean 91 % from 1.5M;
best brain at 1.765M, return 73.0, 96 %).

## Robustness (best brain, `fly eval 50`)

| Test | v9 | **v10** |
|---|---|---|
| clean | 16 % | **90 %** |
| no decoys | 24 % | **90 %** |
| eye noise x3 | 8 % | **96 %** |
| light 2x dimmer or brighter | 20 % | **60 %** |
| block invisible | 0 % | 0 % |

## Precision (`scripts/precision_lift.py`, same rules for both brains)

200 new random layouts each (with decoys):

| Measure | v9 | **v10** |
|---|---|---|
| grasped | 85 % | **100 %** |
| block >= 7.5 cm up at the end | 29 % | **98 %** |
| held within 2.5 cm of the target point at the end | 0 % | **94 %** |
| distance from the target point at the end (median) | 20.4 cm | **1.3 cm** |
| largest sideways drift while in the air (median, lifted episodes) | 23.4 cm | **1.1 cm** |
| time to solve (median) | - | 38 of 100 steps (1.5 s) |

Success map: 5 x 5 block positions across the table, 8 random starts each, decoys removed.
Columns from the robot's right to left, rows near to far. v10 uses the strict rule above; v9 is
shown with its own looser rule (block >= 7.5 cm up at the end), since it scores 0 % on the strict one.

```
v10 (held at target)       v9 (>= 7.5 cm up)
      R           L              R           L
near  6  8  7  8  7        near  0  1  4  1  0
      8  7  7  8  8              0  3  1  1  2
      6  7  6  8  8              2  2  3  5  3
      6  7  8  8  8              0  3  6  2  3
far   6  6  6  8  8        far   0  2  3  3  2
      180/200 (90 %)             52/200 (26 %)
```

The right edge is still the weakest column (6/8), the same side as in reaching.

## Honest notes

* One seed. 50-episode evaluations swing by several points (84-96 % in the last 1.4M steps).
* The teacher used privileged information, but only to make training targets; this is the
  standard teacher -> student recipe for vision-based robot policies. The tested brain sees only
  its eyes, body sense and touch, and fails completely when the block is invisible.
* Practice episodes included help (30 % near-goal starts, 30 % edge/corner blocks); tests,
  maps and videos never do.
* Lighting is the weak spot (60 %): light was never varied in training.
* Imitation should hand the best round (here 7) to PPO, not the last; PPO recovered anyway.

Files: `showcase_best_brain.mp4` (3 solved + 1 missed, new layouts), `progress_2100k_fixed_layouts.mp4`,
`progress_0100k_after_imitation.mp4`, `curves.png`, `eval_results.json`,
`precision_episodes.json` (every precision and map episode, v9 and v10).
