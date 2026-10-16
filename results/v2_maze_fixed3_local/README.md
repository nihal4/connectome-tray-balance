# v2: first maze test (`maze_fixed3`, local, one seed)

A short check of the maze puzzle on a slow 2-core cloud machine (`--profile mac`
settings, 2 environments, 400k steps). Fly-brain network: the 1,000 most connected neurons,
given the ball position and the wall layout as numbers.

| | result |
|---|---|
| solved the fixed maze (eval) | not yet; peak 20 % of eval episodes |
| best model on 20 attempts | 9 reach the goal but cannot hold the ball still for 0.5 s; 11 get stuck 3 cells from the goal |

**Reading:** the network learns the route partly; what it lacks is the fine control to brake
and stop the ball in the goal. `connectome_attempts.mp4` shows one of each (left: reaches the
goal but cannot stop, right: stuck halfway).

Files: `curves.png`, `connectome_attempts.mp4`, and the run folder (best model,
`vecnormalize.pkl`, eval curve, training log, `subset.npz`).
