"""Diagnostic videos for a maze run: from the real start, and with the ball placed
k cells before the goal. Shows where the policy gets stuck and whether it can finish
from close by.

    MUJOCO_GL=egl python scripts/watch_maze.py --run runs/maze_fixed3_vision-connectome-real-s0
    MUJOCO_GL=egl python scripts/watch_maze.py --run ... --before 3 2 1 --best

Videos are written next to --out (default: the run's parent folder) as diag_*.mp4.
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")      # must be set before mujoco is imported
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from ctb.envs import build_env  # noqa: E402
from ctb.envs.labyrinth import BALL_R  # noqa: E402
from scripts.evaluate import add_eye_view, load, make_norm, run_config, run_env_kw  # noqa: E402


def route_cells(env):
    cells, cur = [env.start], env.start
    while cur != env.goal:
        cur = env._next_cell(cur)
        cells.append(cur)
    return cells


def place_ball(env, cell):
    tpos, trot = env._tray_frame()
    x, y = env._cell_center(cell)
    env.data.qpos[env.ball_qadr:env.ball_qadr + 3] = tpos + trot @ np.array([x, y, BALL_R + 0.001])
    env.data.qvel[env.ball_dadr:env.ball_dadr + 6] = 0
    mujoco.mj_forward(env.model, env.data)
    env._phi, env._prev_frame = env._potential(np.array([x, y])), None
    return env._get_obs()


def watch(model, vn, env_name, before, out_dir, seed, robot_kw=None):
    env = build_env(env_name, seed, render_mode="rgb_array", **(robot_kw or {}))
    obs, _ = env.reset(seed=seed)
    model.reset_memory()
    label = "real_start"
    if before is not None:
        obs = place_ball(env, route_cells(env)[-1 - before])
        label = f"{before}_cells_before_goal"
    frames, info, t = [], {}, 0
    for t in range(env.max_steps):
        a, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
        obs, _, term, trunc, info = env.step(a[0])
        frame = env.render()
        if getattr(env, "vision", False):
            frame = add_eye_view(frame, env.camera_frame())
        frames.append(frame)
        if term or trunc:
            break
    path = os.path.join(out_dir, f"diag_{label}.mp4")
    imageio.mimsave(path, frames + [frames[-1]] * 25, fps=50)
    env.close()
    print(f"{label:22s} solved={info.get('is_success')}  steps={t + 1}  "
          f"cells left at the end={info.get('cells_left')}  -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--before", type=int, nargs="*", default=[2, 1],
                    help="also start the ball this many cells before the goal")
    ap.add_argument("--best", action="store_true", help="use the best-eval model instead of the latest")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env_name = run_config(args.run)["env"]
    model = load(args.run, best=args.best)
    vn = make_norm(args.run, {})
    out_dir = args.out or os.path.dirname(os.path.abspath(args.run.rstrip("/")))
    os.makedirs(out_dir, exist_ok=True)
    for k in [None] + list(args.before):
        watch(model, vn, env_name, k, out_dir, args.seed, run_env_kw(args.run))


if __name__ == "__main__":
    main()
