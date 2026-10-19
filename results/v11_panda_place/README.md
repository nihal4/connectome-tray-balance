# v11: Panda pick and place into a bin, by fly eyes (desktop, one seed)

Run: `panda_place-connectome-real-s0-imitate`, stopped at 0.77M steps (levelled off), under 4 hours.
Same 4,000-neuron FlyWire brain, compound eyes, body sense and touch, started from the v10 lift brain.

```
fly job panda_place 3000000 "--init_from panda_lift-v10 --imitate 10 --privileged --grip_mode absolute
  --lift_reward goal --edge_bias 0.3 --random_start 0.3 --std 0.15 --lr 1e-4 --eval_episodes 50"
```

Task: a shallow red bin (14 x 14 cm, 1.5 cm rim) at a new place every episode, at least 15 cm from the
yellow block (0-2 blue decoys). Grasp the block, carry it over the bin, lower it and let go. Solved =
block resting inside the bin (centre within 4.5 cm of the bin centre, on its floor), touched by neither
finger, for 0.2 s, within 150 steps (6 s).

## Training

Teacher rounds (DAgger, 165k steps; `place_teacher` knows the true block and bin positions, training
targets only), student alone after each round: 2, 12, 20, 46, 70, **82**, 88, 78, 84, 72 % solved.
PPO started from the best round by return (round 6) and stayed at 84-92 % on every evaluation
(best brain at 615k: return 98.2, 90 %; last evaluations 92 %).

## Tests (best brain, `fly eval 50`)

| Test | Solved |
|---|---|
| clean | 92 % |
| no decoys | 92 % |
| eye noise x3 | 86 % |
| brightness +-40 % (training +-20 %) | 58 % |
| yellow block invisible | 0 % |

## Precision (`scripts/precision_place.py`)

200 new random layouts: grasped 94 %, carried over the bin 90 %, solved 84 % (the 50-episode test
said 92 %; the true rate is in the high 80s). Block centre to bin centre when placed: median 1.0 cm
(90th percentile 2.9 cm). Time to solve: median 67 steps (2.7 s).

Misses (33 of 200): held over the bin but never lowered (14; the inspected case: block gripped high,
hanging tilted, the arm hovers 2-3 cm above the floor), never grasped (12), grasped but not carried (4),
in the bin but still held (2), other (1).

Success map by BIN position (5 x 5 grid over the bin area, 8 random block/start layouts each):

```
      R           L
near  8  6  5  8  6
      7  5  3  6  6
      7  8  6  7  6
      8  8  7  7  6
far   7  7  7  6  3
      160/200 (80 %)
```

Weakest: the bin near the table centre (3/8) and in the far-left corner (3/8).

## Honest notes

* One seed; 50-episode tests swing by several points (84 % vs 92 % between two samples).
* Practice episodes had help (30 % near-block starts, 30 % edge/corner blocks); tests never do.
* Lighting is the weak spot (58 %), as for the lift.

Files: `showcase_best_brain.mp4` (3 solved + 1 miss), `progress_0700k_fixed_layouts.mp4`,
`progress_0500k_rim_miss.mp4`, `progress_after_teacher.mp4`, `teacher_scripted.mp4` (the scripted
teacher, for comparison), `curves.png`, `eval_results.json`, `precision_episodes.json`.
