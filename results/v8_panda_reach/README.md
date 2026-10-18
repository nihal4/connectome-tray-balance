# v8: Panda arm reaches a block it finds with fly compound eyes (desktop, one seed)

Robot: Franka Panda arm + gripper (MuJoCo Menagerie), hand-space control `[dx, dy, dz, grip]`
at 25 Hz. Brain: 4,000 FlyWire neurons, 46,381 real connections (`flybrain_eye160_N4000.npz`).
Senses: two compound eyes (160 facets each; one on a head beside the table, one on the wrist)
into photoreceptor neurons, body sense into ascending neurons. Task `panda_reach`: bring the
gripper to just above a yellow 5 cm block (new place every episode, 0-2 blue decoys) and
stay within 3 cm for 0.2 s. 60 steps (2.4 s).

## Runs

1. **Main run** `panda_reach-connectome-real-s0`, 1.6M steps, lr 3e-4, `--random_start 0.3`.
   First solves at ~800k; eval peaked at 90 % (10 episodes per eval) around 1.3-1.45M, then
   declined (return 40 -> 25 by 1.6M; action std had shrunk to 0.175). Stopped; the best brain
   (1.3M) was kept.
2. **Fine-tune** `panda_reach-connectome-real-s0-ft`, 1M steps from the 1.3M best brain:
   lr 1e-4, action std reset to 0.25, 50 % of TRAINING blocks placed near the table edges /
   corners (`--edge_bias 0.5`: corners 16 % -> 39 % of episodes), 50 eval episodes per point.
   Eval 80-88 % early, 92-98 % from 700k; best brain at 900k (94 %).

## Where on the table it works (5 x 5 grid of block positions, 6 starts each, no decoys)

| | right edge | | centre | | left edge |
|---|---|---|---|---|---|
| near | 2 -> 3 | 5 -> 6 | 5 -> 6 | 2 -> 6 | 1 -> 5 |
| | 6 -> 6 | 6 -> 6 | 6 -> 6 | 4 -> 6 | 4 -> 6 |
| | 4 -> 5 | 6 -> 6 | 6 -> 6 | 4 -> 6 | 3 -> 6 |
| | 5 -> 5 | 6 -> 6 | 6 -> 6 | 6 -> 6 | 4 -> 6 |
| far | 2 -> 5 | 6 -> 6 | 5 -> 6 | 3 -> 6 | 2 -> 5 |

(main run best brain -> fine-tuned best brain). Whole table: 109/150 (73 %) -> 141/150 (94 %).

Exact corner tips (20 starts each): centre 18 -> 20, far-left 9 -> 15, near-left 6 -> 11,
near-right 5 -> 10, far-right 10 -> 4. Corner misses end ~30 cm away and drift toward the
table centre: the brain does not find a block at the extreme tips (small and dim for the head
eye, outside the downward wrist eye) and falls back to where blocks usually are.

Decoys (main run, 60 episodes each): 0 decoys 82 %, 1 decoy 83 %, 2 decoys 83 %; 1 of 31 misses
ended over a decoy. Decoys are not the problem; table position is.

## Robustness (fine-tuned best brain, 20 episodes each)

| Test | Solved | Return |
|---|---|---|
| clean | 90 % | 35.3 |
| no decoys | 90 % | 34.8 |
| eye noise x3 | 75 % | 28.3 |
| light 2x dimmer or brighter | 70 % | 25.6 |
| yellow block invisible | 5 % | 6.6 |

With the block invisible it fails: it steers by what its eyes see.

## Lessons

* A success rule the policy cannot meet under its own exploration noise (speed < 5 cm/s) stalls
  learning; a 0.2 s dwell works.
* Late PPO training can degrade a good policy; keep the best brain, fine-tune at a lower lr.
* Test videos should show spread-out layouts; ours first showed two near-identical ones.
* Practice where it fails: putting more training blocks at the edges raised the table-wide
  rate from 73 % to 94 %.

Files: `main_1300k_three_solved.mp4` (main run, original three test layouts, all reached),
`main_1500k_corner_miss.mp4` (spread-out layouts: the near-left corner missed), `curves_main_run.png`, `curves_finetune.png`, `eval_results_finetune.json`,
`finetune_200k_all_three_solved.mp4` (three spread-out test layouts, all solved),
`finetune_1000k_latest_brain.mp4` (latest brain, layout 1 missed; the best brain solves it).
