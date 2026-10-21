# v17: greenhouse phase 2a, pick only the ripe tomato (wrist depth camera; desktop, one seed)

Run: `panda_pickripe-connectome-real-s0-cam-1ripe`, stopped at ~270k steps (98 % at 215k and 265k), best brain
at 265k. Started from the v16 tomato-picking brain; 10 teacher rounds (165k steps), then PPO.

```
fly job panda_pickripe 1500000 "--ripe_max 1 --wrist_cam rgbd --init_from panda_harvest-connectome-real-s0-adapt
  --imitate 10 --eye_adapt --privileged --grip_mode absolute --random_start 0.3 --grasp_start 0.2 --std 0.15
  --lr 1e-4 --eval_episodes 50"
```

Task: a truss of three cherry tomatoes side by side (6.5 cm apart); exactly one is ripe, the others green or
turning. Pick the ripe one into the basket, leave the rest. The wrist eye is a close-range depth camera
(D405-like; 16 x 10 pixels of red, nearness and green into the same 480 wrist-eye neurons); the head keeps
the fly eye; smell as before. See the main README for why (three earlier tries) and how the camera works.

Learning: student alone 8 / 10 / 2 % after teacher rounds 1-3, 70-86 % after rounds 4-7, 96-98 % after
rounds 8-10; PPO then 96-98 %.

## How well and how does it choose (`scripts/precision_pickripe.py`, best brain)

| | Layouts | Solved | Ripe tomato into the basket | Unripe tomatoes picked |
|---|---|---|---|---|
| normal | 200 | **98 %** | 195 / 200 | **0 / 400** |
| every tomato looks ripe (camera and fly eye; only smell differs) | 50 | 54 % | 30 / 50 | 20 / 100 (in 38 % of layouts) |
| antennae silenced | 50 | **0 %** | 0 / 50 | 0 / 100 |

It chooses mainly by colour: when all three look ripe it is right about half the time (guessing would be
one in three, so smell adds a little) and often takes an unripe one. Smell is still needed to grasp at all,
as in phase 1: with the antennae silenced it never closes on a tomato. Median time: 74 steps (3.0 s).

## Tests (best brain, `fly eval 50`)

| Test | Solved |
|---|---|
| clean | 92 % |
| eye noise x3 | 90 % |
| brightness +-40 % | 96 % |
| brightness x0.4 or x2.5 | 90 % |
| smell noise x3 | 94 % |
| tinted light (receptor types / colour channels x0.7 / x1.4 / x1.0) | **12 %** |
| uneven light | **0 %** |

(The `fly eval` versions of `target_hidden` and `all_look_ripe`, 96 % and 88 %, ran before those tests were
made to apply to the camera; the table above has the corrected `all_look_ripe`.)

Weak spots: a colour cast (the camera has auto exposure but no white balance, so tinted light shifts the
red/green ratio it judges ripeness by) and uneven light, as in every task so far.

## Files

- `showcase_best_brain.mp4`: the best brain on new layouts, new presentation layout (plus one miss)
- `progress_0200k.mp4`: the fixed test layouts at 200k (old video layout)
- `pickripe_episodes.json`, `eval_results.json`, `curves.png`
