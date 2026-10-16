# v1: TrayBalance on Kaggle (one seed)

Fly-brain network: the 1,000 most connected FlyWire neurons (inputs and outputs picked by
graph structure). Trained on Kaggle CPU, 1M steps, 4 environments, `--profile kaggle`.
Code version: git tag `v1-tray-balance`. The environment changed after this (level home
pose, tray on a post, arm collisions separated from the ball), so this model only behaves
as reported in the v1 environment; a replay is easiest in the Kaggle notebook.

The policy is given the ball position and speed as numbers (no camera).

## Results

Solved the clean task by 100k steps (first eval at 500/500 steps).
Robustness, mean steps survived out of 500 (5 episodes each):

| test | result |
|---|---|
| clean | 500 |
| sensor noise 0.1 | 500 |
| sensor noise 0.3 | 500 |
| blind (ball hidden) | 31 |
| heavy ball 3x | 500 |
| random pushes | 500 |

**Reading:** easy task, learned fast and robust except when the ball is hidden (the network
has no memory). This is why harder tasks (mazes) and the camera version came next.

Files: `tray-connectome-real-s0/` (best model, normalisation, eval curve, log, wiring),
`curves.png`, `connectome_policy.mp4`.
