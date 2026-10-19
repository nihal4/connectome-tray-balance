# v13: colour sorting under changing light (desktop, one seed)

Run: `panda_sort-connectome-real-s0-light`, from the v12 sorting brain, stopped at 0.63M steps.
Practice episodes had brightness anywhere from x0.5 to x2 (`--light_log 0.69`; before: x0.8-1.2);
tests use the normal lighting unless stated.

```
fly job panda_sort 2000000 "--init_from panda_sort-v12 --imitate 10 --light_log 0.69 --privileged
  --grip_mode absolute --lift_reward goal --edge_bias 0.3 --random_start 0.3 --std 0.15 --lr 1e-4 --eval_episodes 50"
```

## Fixed brightness (`scripts/light_sweep.py`, the same 40 new layouts at every level, best brain)

| Brightness | x0.4 | x0.5 | x0.7 | x1 | x1.4 | x2 | x2.5 |
|---|---|---|---|---|---|---|---|
| v12 (practised x0.8-1.2) | 0 % | 0 % | 15 % | 90 % | 0 % | 0 % | 0 % |
| **v13** (practised x0.5-2) | 5 % | 42 % | **92 %** | 82 % | **70 %** | **62 %** | 0 % |

v12 only worked at the brightness it practised under. v13 works across most of its practised range,
weaker towards its edges (x0.5, x2), and not at all beyond it (x0.4, x2.5): it learned the range it saw,
it did not learn to ignore brightness.

## Normal light (300 new layouts, `scripts/precision_sort.py`)

| Block | Solved | Left bin | Centre bin | Right bin | No bin |
|---|---|---|---|---|---|
| yellow (-> left) | 91 % | **94** | 3 | 0 | 6 |
| blue (-> centre) | 86 % | 0 | **93** | 1 | 13 |
| orange (-> right) | 94 % | 0 | 1 | **85** | 4 |

91 % solved (v12: 95 %), 5 of 300 in a wrong bin (v12: 0). The wider lighting cost a little at normal light.

`fly eval 50`: clean 78 %, brightness +-40 % 82 %, eye noise x3 80 %, x0.4 or x2.5 4 %, block hidden 0 %.
The run's own chart showed up to 94-96 %: the best of ~20 noisy 50-episode tests is optimistic, and this run
evaluated its best brain (565k) with the input scaling of 632k (fixed since: `best_vecnormalize.pkl`).

## Next

Real fly photoreceptors adapt to the light level, so the brain mostly receives contrast rather than absolute
brightness. Adding that adaptation to the eye model should remove the dependence on brightness itself,
including beyond any practised range.
