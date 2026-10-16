# v2: first maze test (`maze_fixed3`, local, one seed)

A short check that the maze puzzle is learnable, run on a slow 2-core cloud
machine (`--profile mac` settings, 2 environments, 400k steps each). Same code
as this commit.

| | connectome (1000 fly neurons) | MLP (same size) |
|---|---|---|
| solved the fixed maze | no (peak 20 % of eval episodes) | **yes, 100 % from 275k steps** |
| steps to solve (eval) | – | ~100 (the scripted planner needs ~145) |

Robustness of the MLP (5 episodes each):

| test | solved |
|---|---|
| clean | 100 % |
| sensor noise 0.1 / 0.3 | 0 % / 0 % |
| blind (ball hidden) | 0 % |
| heavy ball 3x | 100 % |
| random pushes | 100 % |
| unseen random mazes | 0 % |

**Reading:**
- The puzzle is learnable with PPO in a few hundred thousand steps (MLP); the video
  shows it rolling the ball around the wall to the goal, not taking a shortcut.
- The MLP memorised this one maze: it fails every unseen maze and breaks under small
  sensor noise. Learning `maze3` (a new maze every episode) is the real puzzle test.
- The connectome policy was clearly slower here. One seed and 400k steps is not enough
  to call it worse: it needs more steps and seeds on Kaggle. This is the first task
  where the two architectures behave differently.

Files: `curves.png`, `mlp_solves_fixed_maze.mp4`, and per run the best model,
`vecnormalize.pkl`, eval curve, training log and (connectome) `subset.npz`.
