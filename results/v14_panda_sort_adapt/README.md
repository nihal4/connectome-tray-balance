# v14: colour sorting with light-adapting fly eyes (desktop, one seed)

Run: `panda_sort-connectome-real-s0-adapt`, from the v12 sorting brain, 1.47M steps (best brain at 1.215M).
New: `--eye_adapt`, photoreceptor light adaptation. Each receptor type (R1-6, R7, R8) of each eye scales its
gain to the average light it currently receives, so the brain gets contrast rather than absolute brightness;
photon noise is added before adaptation and grows like sqrt(light). **Practice used normal light only**
(x0.8-1.2): every other lighting below is something this brain never practised under.

```
fly job panda_sort 1500000 "--init_from panda_sort-v12 --imitate 10 --eye_adapt --privileged --grip_mode absolute
  --lift_reward goal --edge_bias 0.3 --random_start 0.3 --std 0.15 --lr 1e-4 --eval_episodes 50"
```

Teacher rounds alone already reached 72-98 % solved; practice then stayed at 90-100 % on every evaluation.

## Fixed brightness (`scripts/light_sweep.py`, the same 40 new layouts at every level)

| Brightness | x0.4 | x0.5 | x0.7 | x1 | x1.4 | x2 | x2.5 |
|---|---|---|---|---|---|---|---|
| v12 (practised x0.8-1.2) | 0 % | 0 % | 15 % | 90 % | 0 % | 0 % | 0 % |
| v13 (practised x0.5-2) | 5 % | 42 % | 92 % | 82 % | 70 % | 62 % | 0 % |
| **v14 (adapting eyes, practised x0.8-1.2)** | **100 %** | **100 %** | **100 %** | **100 %** | **100 %** | **100 %** | **100 %** |

Expected in large part: a change of overall brightness is cancelled by adaptation (eye output differs by < 0.3 %),
up to the extra photon noise in dim light.

## What adaptation cannot simply cancel

Uneven light: one side of each eye's view brighter, the other dimmer (40 layouts each).

| Uneven light | x1.2 / x0.83 | x1.4 / x0.7 | x2 / x0.5 |
|---|---|---|---|
| v12 | | | 3 % |
| v13 | | | 3 % |
| **v14** | **93 %** | **88 %** | **48 %** (60 % in `fly eval`) |

Tinted light (receptor types x0.7 / x1.4 / x1.0): v12 15 %, v13 30 %, **v14 98 %** (each receptor type adapts on
its own, which also corrects a colour cast, like colour constancy in animals and cameras).

## Normal light (300 new layouts, `scripts/precision_sort.py`)

| Block | Solved | Left | Centre | Right | No bin |
|---|---|---|---|---|---|
| yellow (-> left) | 98 % | **101** | 0 | 0 | 2 |
| blue (-> centre) | 97 % | 0 | **104** | 0 | 3 |
| orange (-> right) | 100 % | 0 | 0 | **90** | 0 |

**98 % solved, no block in a wrong bin** (v12: 95 %, v13: 91 % with 5 wrong). Median 2.4 s to sort, block lands
1.3 cm from the bin centre.

`fly eval 50`: clean 98 %, eye noise x3 98 %, brightness +-40 % 96 %, x0.4 or x2.5 100 %, uneven light 60 %,
tinted light 98 %, block hidden 0 %.

## Honest notes

* One seed. The light model is a global multiplier plus these two tests; real lighting also has shadows,
  reflections and direction, which are not modelled.
* Strong uneven light (x2 / x0.5 across the view) is the remaining weak spot (48-60 %).
* Showcase miss: a yellow block placed just in front of the bins; the arm never went down to it.

Files: `showcase_best_brain.mp4` (3 solved + 1 miss), `progress_1400k_fixed_layouts.mp4`, `curves.png`,
`eval_results.json`, `light_sweep.json`, `sort_episodes.json`.
