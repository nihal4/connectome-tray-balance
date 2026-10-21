# v16: greenhouse phase 1, pick a tomato off the plant (desktop, one seed)

Run: `panda_harvest-connectome-real-s0-adapt`, stopped at ~607k of 1.5M steps (solved 100 % of the 50 test
episodes at 315k and 565k; the curves were flat), best brain at 565k. Started from the v15 smell brain
(4,476 neurons, the FlyWire eye, body and smell pathways), first copying a scripted teacher (10 DAgger
rounds, 165k steps), then PPO.

```
fly job panda_harvest 1500000 "--init_from panda_smell-connectome-real-s0-adapt --imitate 10 --eye_adapt --privileged
  --grip_mode absolute --random_start 0.3 --grasp_start 0.2 --std 0.15 --lr 1e-4 --eval_episodes 50"
```

Task: one cherry tomato (3.2 cm, 15 g) hangs on its stalk from a tomato plant somewhere along the far side
(13-25 cm high). The hand comes from the front and above (fingers tilted 30 deg towards the plant), closes on
the tomato and pulls it off (the stalk snaps when the held tomato is pulled or bent with > 1.5 N), carries it
to the basket beside the robot and drops it in. Touching the plant costs reward. The tomato smells ripe.
Solved = resting in the basket, released, within 8 s. See the main README for the physics.

Learning: student alone 0 % after the first teacher round, 26 / 78 / 84 % after rounds 2-4, 96 % after round
10; PPO then 98-100 % (100 % at 315k and 565k).

## How well, how fast, how gently (`scripts/precision_harvest.py`, best brain, 200 new layouts)

| | Result |
|---|---|
| Solved (tomato in the basket) | **98 %** |
| Picked off the plant | 99 % |
| Dropped on the way | 1 % |
| Time to pick (median) | 28 steps = **1.1 s** |
| Time to the basket (median) | 59 steps = **2.4 s** (the scripted teacher: 76 steps = 3.0 s) |
| Robot touched the plant | in 90 % of episodes, for 5.8 steps (0.23 s) on average: brief brushes of the stalk while closing (the teacher: 3.6 steps) |

## What does it use smell for? (50 layouts each)

| Smell | Solved | Picked | Closest the hand got to the tomato |
|---|---|---|---|
| normal | 98 % | 99 % | 0.5 cm |
| antennae silenced | **0 %** | 0 % | 4.1 cm (stops short, never closes) |
| comes from 6 cm beside the tomato | 96 % | 98 % | 0.6 cm (still goes to the real tomato) |
| full strength everywhere (no rise when close) | **20 %** | 36 % | 13.5 cm |

It finds the tomato by sight (moving the smell source 6 cm to either side does not move the hand), but uses
how strong the smell gets as the hand closes in to judge "close enough, grasp now": with no smell it stops
4 cm short and never closes; with full-strength smell everywhere it grabs too early or at nothing. Flies also
use rising odour concentration when they approach food. With the tomato hidden from the eyes it still solves
90 % (`fly eval`): the visible stem and branch show where the tomato hangs.

## Tests (best brain, `fly eval 50`)

| Test | Solved |
|---|---|
| clean | 98 % |
| eye noise x3 | 94 % |
| brightness +-40 % | 98 % |
| brightness x0.4 or x2.5 | 94 % |
| tinted light | 96 % |
| tomato hidden from the eyes | 90 % |
| smell noise x3 | 98 % |
| antennae silenced | 0 % |
| uneven light (one side of the view x2, other x0.5) | **0 %** |

Strongly uneven light is the weak spot again, worse than for sorting (v14: 48-60 %): the hand wanders off to
the edge of its reach. Greenhouse light under leaves is uneven, so phase 4 (leaves, shadows) has to train
with it.

## Files

- `showcase_best_brain.mp4`: the best brain on new layouts (successes and one drop)
- `progress_0500k.mp4`: the three fixed test layouts at 500k
- `teacher_scripted.mp4`: the scripted teacher
- `harvest_episodes.json`: every episode of the tables above; `eval_results.json`; `curves.png`

The plant in the videos is a full greenhouse tomato plant with neighbours; that is drawn for people only.
The fly eyes and the physics see the working parts (stem, branch, stalk, three leaves, tomato).
