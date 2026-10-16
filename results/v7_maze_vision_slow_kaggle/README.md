# v7: camera maze solved, with slow motors (Kaggle, one seed)

Run: `maze_fixed3_vision-connectome-real-s0-slow`, 2M steps, fly-robot brain (2,500 neurons),
`--action_rate 0.1 --random_start 0.5`. Same brain, reward and success rule as v6; only the
motors are slower and half of the training episodes start partway along the route.

Learning: first solve at 350k steps, 70 % at 750k, 100 % at 1.1M, 100 % on every eval from
1.65M to 2M. Episode length kept falling to ~130 steps (about 2.6 s of the 6 s allowed).

Robustness (10 episodes each, best model):

| Test | Solved | Return | Steps |
|---|---|---|---|
| clean | 100 % | 35.7 | 132 |
| heavy ball (3x) | 100 % | 35.0 | 141 |
| random pushes | 60 % | 24.2 | 220 |
| camera noise 0.1 | 0 % | 0.0 | 300 |
| camera noise 0.3 | 0 % | -2.8 | 300 |
| ball hidden from camera | 0 % | -1.9 | 300 |
| unseen mazes | 0 % | -8.2 | 300 |

Reading: it solves the maze it trained on quickly and stops the ball in the goal centre, copes
with a heavier ball and most pushes, and really steers by the camera (fails when the ball is
hidden). It has memorised this one maze: 0 % on new mazes. Noisy pixels, never seen in
training, break it (same as the tray brain in v5).

Files: `maze_fixed3_vision_clean.mp4` (136 steps, solved), `curves.png`, `eval_results.json`.
