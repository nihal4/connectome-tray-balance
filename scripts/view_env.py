"""Open the MuJoCo viewer and watch the arm (PD controller, random, or a trained run).

On macOS the passive viewer must be started with `mjpython` (installed with mujoco):
    mjpython scripts/view_env.py --mode scripted
    mjpython scripts/view_env.py --env maze4 --mode scripted
    mjpython scripts/view_env.py --mode policy --run runs/maze3-connectome-real-s0
On Linux/Windows plain `python` works.

Runs at real-time speed (50 Hz), so it is light on the CPU.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ctb.safety import limit_threads  # noqa: E402

limit_threads(1)

import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

from ctb.envs import build_env, scripted_controller  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="tray")
    ap.add_argument("--mode", choices=["scripted", "random", "zero", "policy"], default="scripted")
    ap.add_argument("--run", default=None)
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()

    if args.mode == "policy":
        from scripts.evaluate import load, make_norm, run_config
        args.env = run_config(args.run)["env"]
    env = build_env(args.env, seed=0)
    act = {"scripted": lambda o: scripted_controller(env, o), "zero": lambda o: np.zeros(5),
           "random": lambda o: env.action_space.sample()}.get(args.mode)
    if args.mode == "policy":
        model = load(args.run)
        vn = make_norm(args.run, {})
        act = lambda o: model.predict(vn.normalize_obs(o[None]), deterministic=True)[0][0]  # noqa: E731

    obs, _ = env.reset()
    dt = env.model.opt.timestep * env.frame_skip
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=ep)
            for t in range(env.max_steps):
                t0 = time.time()
                obs, r, te, tr, info = env.step(act(obs))
                viewer.sync()
                if not viewer.is_running():
                    return
                time.sleep(max(0, dt - (time.time() - t0)))
                if te or tr:
                    break
            print(f"episode {ep}: {t + 1} steps, success={info.get('is_success', not info.get('fell'))}")


if __name__ == "__main__":
    main()
