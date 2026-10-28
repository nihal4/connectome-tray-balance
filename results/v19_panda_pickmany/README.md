# v19: greenhouse phase 2c, any truss (2-6 tomatoes, 1-3 ripe; desktop, one seed)

Task (`panda_pickmany`): a truss of 2 to 6 cherry tomatoes on irregular stalks along a zig-zag stem, 1 to 3 of
them ripe, the rest green or turning. Pick every ripe one into the basket, leave the rest, bruise nothing.
Solved = all ripe tomatoes in the basket, nothing unripe picked, no tomato bruised (gentle handling, as in v18:
gripper force capped at 3 N; any robot contact above 5 N or a landing faster than 1 m/s bruises a tomato).
16 s per truss.

Final brain: `panda_pickmany-connectome-real-s0-2cam-mem-strict-events` (best brain). Same robot and fly brain as
v18 (wrist + front depth cameras, smell, memory); three runs led to it:

```
# 1. from the v18 brain: teacher rounds, then PPO (4M steps; full options in the run's config.json)
                                                                                        -> ...-2cam-mem
# 2. a bruise ends the episode (safe-RL style), from its latest brain (2M)
fly job panda_pickmany 2000000 "--init_from panda_pickmany-connectome-real-s0-2cam-mem --init_last --bruise_end
  --lr 5e-5 --return_start 0.3 --eval_episodes 20"                                      -> ...-2cam-mem-strict
# 3. each tomato is paid once, when it is put in the basket (2M)
fly job panda_pickmany 2000000 "--init_from panda_pickmany-connectome-real-s0-2cam-mem-strict --init_last
  --bruise_end --reward events --lr 5e-5 --return_start 0.3 --eval_episodes 20"           -> ...-strict-events
```

## Result (`scripts/precision_pickripe.py`, the same 200 new layouts for every brain)

| Brain | Solved | 1 ripe | 2 ripe | 3 ripe* | Ripe in basket | Unripe picked | Bruised |
|---|---|---|---|---|---|---|---|
| 1. teacher + PPO, latest (4M) | 60 % | 71 % | 60 % | ~42 % | 324 / 373 | 4 % of layouts | 28 % of layouts |
| 2. + a bruise ends the episode, latest | 62 % | 78 % | 65 % | ~31 % | 292 / 373 | 3 of 227 | 15 % |
| 3. + each tomato paid once, latest | 62 % | 81 % | 54 % | ~47 % | 296 / 373 | **0** of 227 | **11 %** |
| **3. + each tomato paid once, best (final)** | **67 %** | 81 % | 64 % | **~51 %** | **321 / 373** | 6 of 227 | 14 % |

\* 45 of the 200 layouts have three ripe tomatoes; derived from the totals. The scripted teacher itself solves
75-83 % of these trusses, so the student is within ~10 points of what it was shown. Median time 5.6 s.

Senses (final brain, 50 layouts each): antennae silenced 0 % (smell still times the grasp); every tomato looking
ripe 44 %, unripe picked in 48 % of layouts (colour decides which); turning tomatoes made to smell ripe 56 %,
unripe picked in 12 % (it checks the colour).

## What each step fixed, and how it was found

1. **Bruising** (run 1: 28 % of layouts). With a bruise only costing reward, PPO traded bruises for speed.
   `--bruise_end`: the first bruise ends the episode with -1, a terminal safety violation as in safe RL. Bruised
   layouts 28 % -> 15 %; solved unchanged, because something else took the bruises' place:
2. **Quitting after the first tomato** (run 2: 33 of 76 failures, 12 before). `scripts/failures_pickripe.py`
   followed the hand after its last drop into the basket: it dropped the first tomato at step ~76, then stayed
   away from the truss for the remaining ~18 s at 9 cm/s, and never once tried to grasp the ripe tomato left.
   More time changed nothing (520 instead of 400 steps: 62 % -> 62 %). The reason was the reward: every tomato
   already in the basket paid ~0.5 per step for the rest of the episode, and with a bruise ending the episode,
   every risky move near the truss put all of that at stake; stopping after one tomato became the safe choice.
   `--reward events` pays the change of the same progress measure, so a tomato in the basket is banked and
   cannot be lost; finishing the harvest pays the rest (here, with the inherited `--smooth`, the remaining
   time's reward in one lump) and ends the episode; a bruise or an unripe pick costs -1 and ends it; standing
   back earns nothing. Three-ripe trusses ~31 % -> ~51 %, 29 more ripe tomatoes in the basket, solved 67 %.

## Where it still fails (final brain, `scripts/failures_pickripe.py`, same 200 layouts)

| Cause | Layouts |
|---|---|
| bruised a tomato | 25 |
| a later ripe one never picked | 20 |
| never picked the first ripe one | 8 |
| ripe one picked but not in the basket | 7 |
| picked an unripe one | 5 |
| tomato flung | 1 |

First bruises: mostly the fingers closing on the ripe tomato being picked (10) or knocking a neighbour on the
way in (9), and the tomato held while carrying (6). "A later ripe one never picked" is down from 31 to 20, but
it is the same behaviour: the hand stays back (93 % of the remaining time) and never tries. Both are limits of
this phase: the teacher's own approach is not gentle enough to show a better one, and the brain does not
always find the next tomato again.

## Also found on the way

- **Physics bug**: the unused blue blocks of the older tasks, parked under the table, were shot up at ~300 m/s
  by the infinite floor plane at every reset (12 deep contacts in the first steps). Parked blocks no longer
  collide and float. The teacher solved 30/40 test trusses after the fix, 27/40 before.
- **GPU training was measured, not assumed**: MuJoCo MJX on a Kaggle T4 stepped this scene at 30 env
  steps/s for 64 arms and 33 for 256 (out of memory at 1024: 2854 constraint rows per arm), slower than the
  training PC (55 with the fly brain and PPO). Training stays on the CPU.

## Files

- `showcase_2000k.mp4`: the final brain on new trusses (not the 200 test layouts): three solved, then one miss
  (red frame); insets show both cameras, what the brain receives from each, and the smell at the hand
- `curves.png`: the last fine-tune (20-layout tests swing by +-15 points; the 200-layout test is the measure)
- `precision_summary.json`: every number above (all brains, senses tests, failure counts)
- `config_final.json`: the final run's settings
