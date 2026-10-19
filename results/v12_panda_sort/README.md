# v12: Panda sorts blocks by colour (desktop, one seed)

Run: `panda_sort-connectome-real-s0-imitate`, stopped at 0.515M steps (~3 h). Same 4,000-neuron
FlyWire brain, compound eyes, body sense and touch, started from the v11 pick-and-place brain.

```
fly job panda_sort 3000000 "--init_from panda_place-v11 --imitate 10 --privileged --grip_mode absolute
  --lift_reward goal --edge_bias 0.3 --random_start 0.3 --std 0.15 --lr 1e-4 --eval_episodes 50"
```

Task (a sorting station): one block per episode, yellow, blue or orange at random, on the near half of
the table; three identical red bins in a row on the far side. Yellow -> left, blue -> centre, orange ->
right; solved = resting in its own bin, released, 0.2 s, within 6 s. Colours as the fly's photoreceptors
see them (brightness, UV, green): yellow (1, 0.05, 1), blue (0.75, 0.9, 0.15), orange (0.95, 0.05, 0.2).

## Training

Teacher rounds (student alone, solved): 18, 38, 24, 50, 64, 62, 44, 78, 62, 38 %; PPO started from the
best round by return (round 8, 78 %) and reached 76 % at 215k, 90-98 % from 265k (best: 98 % at 515k).

First design, dropped: bins coloured like the blocks, at random places. The student solved 0 % and got
worse every round (imitation loss 0.04 -> 0.16): there was always a big bin of the block's own colour,
so "find the yellow thing" stopped finding the block.

## Is it really sorting by colour? (`scripts/precision_sort.py`, 300 new layouts)

| Block | n | Solved | Left bin | Centre bin | Right bin | No bin |
|---|---|---|---|---|---|---|
| yellow (-> left) | 103 | 98 % | **101** | 0 | 0 | 2 |
| blue (-> centre) | 107 | 92 % | 0 | **98** | 0 | 9 |
| orange (-> right) | 90 | 97 % | 0 | 0 | **87** | 3 |

95 % solved overall and **no block ever went into a wrong bin**. Misses (14): dropped on the way (6),
still held at the end (5), never grasped (3). Blue, the decoy colour it was taught to avoid earlier, is
the weakest. Placed blocks end a median 1.3 cm from the bin centre (90th pct 2.6 cm); median 69 steps
(2.8 s) to solve.

## Tests (best brain, `fly eval 50`)

| Test | Solved |
|---|---|
| clean | 92 % |
| no decoys (there are none in this task) | 92 % |
| eye noise x3 | 84 % |
| brightness +-40 % (training +-20 %) | 50 % |
| block invisible | 0 % |

## Honest notes

* One seed. The bins are in fixed places (+-1.5 cm), so where each colour goes is a learned rule,
  not something read off the bins.
* Lighting is the weak spot (50 %), as for lift and pick-and-place.

Files: `showcase_best_brain.mp4`, `progress_0500k_fixed_layouts.mp4` (yellow, blue, orange),
`progress_after_teacher.mp4`, `teacher_scripted.mp4`, `curves.png`, `eval_results.json`,
`sort_episodes.json` (all 300 test episodes).
