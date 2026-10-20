# v15: ripe or rotten, by smell alone (desktop, one seed)

Run: `panda_smell-connectome-real-s0-adapt`, 1.26M steps (power cut; it had levelled off at 90-100 % from
~0.1M), best brain at 1.215M. The brain was **grown** from the v14 sorting brain: all 4,000 neurons and their
learned connection strengths kept, plus the fly's smell pathway from FlyWire (106 olfactory receptor neurons,
53 glomerulus types x 2 antennae, + 370 pathway neurons: 164 projection neurons, 128 antennal-lobe local
neurons, lateral-horn and other neurons) = 4,476 neurons. Receptor responses to each chemical: DoOR
(Muench & Galizia 2016, CC BY-SA 4.0). Two antennae beside the gripper; see the main README for the model.

```
fly job panda_smell 1500000 "--init_from panda_sort-v14 --imitate 10 --eye_adapt --privileged --grip_mode absolute
  --lift_reward goal --edge_bias 0.3 --random_start 0.3 --std 0.15 --lr 1e-4 --eval_episodes 50"
```

Task: one fruit (the yellow block) per episode, ripe or rotten at random, looking exactly the same; two identical
bins, ripe -> left, rotten -> right; solved = resting in its own bin, released, within 6 s.

## Does it decide by smell? (`scripts/precision_smell.py`)

| | Fruit | n | Solved | Ripe bin | Rotten bin | No bin |
|---|---|---|---|---|---|---|
| **with smell** (300 layouts) | ripe | 167 | 99 % | **166** | 0 | 1 |
| | rotten | 133 | 97 % | 0 | **129** | 4 |
| **antennae silenced** (100 layouts) | ripe | 62 | 97 % | 60 | 0 | 2 |
| | rotten | 38 | **0 %** | 37 | 0 | 1 |

With smell: 98 % solved, **no fruit in the wrong bin**. Without smell it still grasps and carries every fruit,
but always to the ripe bin: "ripe" is its default, and rot (driven mostly by the acid-sensing receptors DC4,
DP1l, DP1m) is what sends a fruit to the other bin. So the choice of bin comes from smell, and nothing else in
the scene gives it away.

## Tests (best brain, `fly eval 50`)

| Test | Solved |
|---|---|
| clean | 100 % |
| eye noise x3 | 100 % |
| brightness +-40 % | 100 % |
| brightness x0.4 or x2.5 | 98 % |
| tinted light | 100 % |
| uneven light (x2 / x0.5) | 48 % |
| smell noise x3 | 100 % |
| antennae silenced (`no_smell`) | 50 % (chance) |
| fruit invisible | 0 % |

Teacher rounds alone already reached 82-100 %: the grown brain kept every arm skill and only had to learn to
read its new sense.

## Honest notes

* One seed. Fruit recipes are simplified (main chemicals only); 145 of 424 receptor x chemical pairs were never
  measured and count as 0. The odour model is a smooth fall-off with flicker, not a real turbulent plume.
* The mushroom body (odour learning) is not on the shortest pathways and is not included.
* Uneven light is still the weak spot of the eyes.

Files: `showcase_best_brain.mp4` and `progress_fixed_layouts.mp4` (with the fruit label and live antenna panel),
`progress_0100k_after_teacher.mp4`, `teacher_scripted.mp4`, `curves.png`, `eval_results.json`, `smell_episodes.json`.
