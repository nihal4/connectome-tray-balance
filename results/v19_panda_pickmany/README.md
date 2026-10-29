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

## After v19: how far does the fly brain reach on this task? (closing experiments)

Goal set afterwards: a scripted teacher near 100 % and a fly brain at 95 %+. The teacher got there; the brain
did not. Every number below is measured, one seed, same robot and fly brain (4,476 neurons, wiring fixed).

### 1. The teacher: 75-83 % -> 98.0 % (`scripts/teacher_check.py`, 1,000 new random trusses)

Failure taxonomy first, then one fix per cause: a servoed final approach aimed at where the tomato hangs at
rest (chasing a swinging tomato made the hand dither), miss detection and retry, a free spot in the basket and
capped carry speed, backing out before rising when the hand starts below the fruit, and the wrist singularity
(with the hand tilted 60 deg, joints 5 and 7 turn about nearly the same axis; a null-space pull on them only
near the singularity). The task changed slightly with it: the last 3 cm of each branch no longer collide,
500 steps per truss, and a feasibility filter (every ripe tomato reachable from the approach line; it rejected
0 of 1,000 random trusses, so the reported rates are unfiltered). The v19 brain on this version: **72 %** of
the 200 layouts (one / two / three ripe 75 / 80 / ~53 %), 336 / 373 ripe in the basket, bruised 16 %.

### 2. Teacher -> student imitation: never above 26 % (the brain alone, 50 new trusses per test)

Short DAgger runs from the v19 brain, same settings, only the listed change:

| Variant | Round 1 | Round 2 | Round 3 | Result |
|---|---|---|---|---|
| V0: as before | 0 % | 10 % | 0 % | the student freezes at the tomato |
| V1: + per-neuron time constant (`--neuron_tau`, Lappalainen et al. 2024) | 2 % | 14 % | 6 % | null: its parameters barely moved (round-1 losses identical to V0 to 4 digits) |
| V2: + all 156 descending neurons as outputs (`--all_dn`, not 64) | 0 % | 20 % | 0 % | null: same loss as V0 |
| V3: + gripper decisions weighted, student drives 50-step stretches | **14 %** | **26 %** | 16 % | first real gain (round 1, same data as V0: 7/50 vs 0/50) |
| V4: V3 + release weighted, 6 rounds | 4 % | 8 % | 24 % | round 4: 2 %; the loss rose every round |

What the diagnostics showed (`scripts/diag_close_cue.py`, `scripts/diag_student_vs_teacher.py`):
- **The information is in the brain's inputs.** A gradient-boosted tree reads the teacher's "close now" moment
  from the same 1,883 brain inputs with AUC 0.998 (wrist camera alone 0.995) and the hand-to-tomato distance
  to 0.3 mm (median), on trusses it never saw. Depth and resolution are not the limit.
- **The student hovered with the tomato between its fingers.** Running the V2 student and asking the teacher
  at every step: in 41 % of all steps the teacher said CLOSE and the student answered +0.18 (open), the
  average of data in which closing an open hand happens on one step per grasp. Weighting those rare gripper
  decisions (V3) removed the hover (41 % -> 0.1 % of steps) and raised the student from 5/40 to 17/40.
- **What remained** (V3, 23 failures of 40): holding the tomato over the basket while the teacher said "open"
  (8; weighting the release too did not help, V4), and small, undirected arm moves in situations the student
  caused itself (|command| 0.1-0.2 where the teacher said ~1).
- Imitation never reached the brain's own reward-trained level (72 %): copying the teacher pulled it away
  from what it already did, and the more the student drove, the less of the data it could fit.

### 3. Teacher-guided reward training (Jump-Start RL, Uchendu et al. 2023): no gain

`--guide 200`: in each training episode the teacher drives a random 0-200 steps (at most until the last ripe
tomato is picked), then the brain takes over and is paid only for what it does; the teacher's share falls to 0
over training. From the v19 brain, stopped at 1.0M of 2M steps: 20-truss tests averaged ~70 % over the
first 0.5M (the starting level) and ~58 % over the second, best single test 85 %.

### Conclusion

On phase 2c the fly brain reaches **67-72 %** and stays there: plain reward fine-tuning, five imitation
variants (best 26 %), two biologically grounded brain changes (no effect) and teacher-guided practice all
end at or below it. The teacher, with true positions, solves 98 %; the brain's cameras carry the same
information, and a generic learner reads it from them. The limit is how much of the teacher's step-by-step
decisions this connectome-shaped network learns, not what it senses: a measured edge of how far the fly
brain reaches. The arm series ends here; phase 3 (rotten fruit to a reject bin) would add a third decision
to exactly what it finds hard.

## Files

- `showcase_2000k.mp4`: the final brain on four new trusses (not the 200 test layouts): two solved (green
  frame), two missed (red): a three-ripe truss lost early, and a two-ripe truss where the hand hesitates at the
  turning tomatoes; insets show both cameras, what the brain receives from each, and the smell at the hand
- `curves.png`: the last fine-tune (20-layout tests swing by +-15 points; the 200-layout test is the measure)
- `precision_summary.json`: every number above (all brains, senses tests, failure counts)
- `config_final.json`: the final run's settings
- `closing_experiments.json`: the numbers of the closing experiments above
