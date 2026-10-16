# v5: fly-robot brain balancing by vision, full Kaggle run (one seed)

Same setup as v4 (`scripts/build_flybrain.py`, 2,500 neurons wired from the right eye's
photoreceptors to 64 descending command neurons; `tray_vision`: the brain only sees a 16x16
wrist-camera image and feels its joints). Trained on Kaggle CPU with `--profile kaggle`
(4 environments) for 1M steps, about 2.7 hours.

## Results

- **Learning:** about 70 steps survived at 50k, 160 at 150k, then the full 500 steps (10 s)
  from **250k** onward, and it stayed there to 1M.
- **Robustness**, 10 new episodes each:

| test | kept the ball for 10 s | mean steps |
|---|---|---|
| clean | 100 % | 500 |
| ball 3x heavier | 100 % | 500 |
| random pushes on the ball | 100 % | 500 |
| camera noise 0.1 | 0 % | 144 |
| camera noise 0.3 | 0 % | 63 |
| ball removed from the camera image | 0 % | 28 |

**Reading:** this confirms the local v4 result on a full run. The brain balances by what it
sees (it fails at once when the ball disappears from the image), copes with heavier balls and
shoves, and is fragile to pixel noise: the noise test adds independent noise to all 512 eye
inputs every step, which the brain never saw during training. Training with a little camera
noise is the obvious fix to try.

## Videos

- `tray_vision_clean.mp4`: normal run
- `tray_vision_heavy_ball.mp4`: ball 3x heavier (drawn dark grey)
- `tray_vision_push.mp4`: random shoves; the frame turns red on each push

The camera image the brain receives is shown in the top-left corner of each video.

Files: `curves.png`, the three videos, and `tray_vision-flyrobot-s0/` (best model,
normalisation, eval curve, training log, `subset.npz` = the brain's wiring).
