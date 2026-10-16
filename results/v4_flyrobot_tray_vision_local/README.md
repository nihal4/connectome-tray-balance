# v4: fly-robot brain balances the ball by vision (local, one seed)

First test of the fly-robot mode (`scripts/build_flybrain.py`, `tray_vision`), run on a
2-core cloud machine, `--profile mac` settings, stopped at ~310k steps.

**The brain** (2,500 neurons, 32,595 real connections): 512 photoreceptors of the right eye
(R8 = current camera frame, R7 = previous frame), 64 ascending neurons (joint sense),
64 descending neurons (arm commands), and the neurons on the shortest pathways between
them: optic lobe 26 %, visual projection 14 %, central brain 30 %. Every eye-to-command path
is 3-5 synapses long. Only the connection strengths are trained.

**The task:** keep the ball on a flat tray for 10 s, seeing it only through a 16x16 wrist
camera. The brain is not told where the ball is.

| | result |
|---|---|
| learning | 40-100 steps survived until 180k, then jumped to the full 500 at 200k–220k |
| best model, 8 new episodes | **100 % kept the ball for the full 10 s** |
| heavy ball 3x / random pushes | 100 % / 100 % |
| camera noise 0.1 / 0.3 | 0 % / 0 % (the brain is sensitive to pixel noise) |
| blind (ball removed from the camera image) | 0 %, which shows it really steers by what it sees |

The video shows the camera image the brain receives in the top-left corner. The policy keeps
the ball on the tray but not always in the centre; more training should tighten that.

Files: `curves.png`, `flyrobot_balances_by_vision.mp4`, and the run folder (best model,
normalisation, eval curve, training log, `subset.npz` = the brain's wiring).
