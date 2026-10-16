# v1: TrayBalance on Kaggle (one seed)

Trained on Kaggle CPU, 1M steps each, 4 environments, `--profile kaggle`.
Code version: git tag `v1-tray-balance`. The environment changed after this
(level home pose, tray on a post, arm collisions separated from the ball), so
these models are kept as a record. They only behave as reported in the v1
environment (`git checkout v1-tray-balance`); the v1 code also expects the wiring
file at the original Kaggle path, so a replay is easiest in the Kaggle notebook.
The robustness numbers below were produced that way.

## Results

Both runs solved the clean task by 100k steps (first eval at 500/500).
Robustness, mean steps survived out of 500 (connectome: 5 episodes, MLP: 10):

| test | connectome (1000 fly neurons) | MLP (same size) |
|---|---|---|
| clean | 500 | 500 |
| sensor noise 0.1 | 500 | 500 |
| sensor noise 0.3 | 500 | 500 |
| blind (ball hidden) | 31 | 67 |
| heavy ball 3x | 500 | 500 |
| random pushes | 500 | 500 |

**Reading:** with enough training both policies are equally robust, and both fail
when the ball is hidden (expected: neither has memory). The earlier "connectome is
more robust to noise" impression from the 100k-step test run does not hold. The task is
too easy to separate the two architectures, which is why the maze puzzles were added.

## Files

- `tray-*/best_model.zip`, `vecnormalize.pkl`: trained policy and observation normalisation
- `tray-*/evaluations.npz`, `progress.csv`: eval curve and training log
- `tray-*/eval_results.json`: numbers above
- `tray-connectome-real-s0/subset.npz`: the 1000-neuron wiring the model was built on
- `curves.png`, `connectome_policy.mp4`
