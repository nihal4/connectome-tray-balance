# v3: Kaggle maze runs (1M steps each, one seed)

Fly-brain network (1,000 most connected neurons, maze layout given as numbers). From
`train_log.txt` (the notebook's filtered training output). Eval = 10 episodes,
deterministic policy, every 50k steps. `maze3h` was still running when this was saved.

| level | result |
|---|---|
| `maze_fixed3` (one maze) | first success at 300k; 30–40 % solved at 950k–1M, still rising |
| `maze3` (new maze every episode) | 0 % throughout |

**Reading:**
- On the fixed maze the network does learn, slowly: success appears after ~700k steps and
  was climbing at the end.
- Random mazes are not learned in 1M steps.

**Reward hack (fixed after this run):** the maze gave a flat +0.2 per step while the ball
rested in the goal, worth more than finishing, so a policy could hover in and out of the goal
to collect reward. The hold reward is now progress-based (+0.4 per step held, all taken back
if the ball leaves). Runs from this version are not directly comparable with later runs.
