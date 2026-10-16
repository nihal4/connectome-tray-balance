"""Evaluate a trained run, optionally under robustness tests, and record a video.

Examples:
    python scripts/evaluate.py --run runs/tray-connectome-real-s0
    python scripts/evaluate.py --run runs/maze3-connectome-real-s0 --robustness
    python scripts/evaluate.py --run runs/maze3-connectome-real-s0 --episodes 1 --video maze.mp4

Robustness tests (none of these were seen in training):
    noise_0.1/0.3  Gaussian noise on every sensor
    blind          ball position/velocity hidden from the policy ("vision loss")
    heavy_ball     ball 3x heavier
    push           random pushes on the ball
    random_mazes   (only for runs trained on maze_fixed3) unseen mazes

Results are merged into <run>/eval_results.json (one key per test), so running
this again for a video does not wipe earlier robustness numbers.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ctb.safety import limit_threads  # noqa: E402

limit_threads(2)

import numpy as np  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from ctb.envs import build_env  # noqa: E402
from ctb.policies import load_ppo  # noqa: E402

TESTS = {
    "clean": {},
    "noise_0.1": {"obs_noise": 0.1},
    "noise_0.3": {"obs_noise": 0.3},
    "blind": {"ball_blind": True},
    "heavy_ball": {"ball_mass_scale": 3.0},
    "push": {"push_prob": 0.05},
}


def run_config(run):
    cfg = json.load(open(os.path.join(run, "config.json")))
    cfg.setdefault("env", "tray")          # runs from before the maze update
    return cfg


def load(run, best=True):
    best_path = os.path.join(run, "best_model.zip")
    path = best_path if best and os.path.exists(best_path) else os.path.join(run, "model.zip")
    return load_ppo(run, path)


def make_norm(run, env_kw, seed=0):
    env_name = run_config(run)["env"]
    venv = DummyVecEnv([lambda: build_env(env_name, seed, **env_kw)])
    vn = VecNormalize.load(os.path.join(run, "vecnormalize.pkl"), venv)
    vn.training = False
    vn.norm_reward = False
    return vn


def run_episodes(model, run, env_kw, episodes, seed=123):
    rets, lens, succ = [], [], []
    for ep in range(episodes):
        vn = make_norm(run, env_kw, seed + ep)
        obs = vn.reset()
        done, ret, n, info = False, 0.0, 0, {}
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, d, infos = vn.step(a)
            ret += float(r[0]); n += 1; done = bool(d[0]); info = infos[0]
        rets.append(ret); lens.append(n)
        succ.append(float(info.get("is_success", not info.get("fell", False) and n >= 500)))
        vn.close()
    return {"return": round(float(np.mean(rets)), 1), "length": round(float(np.mean(lens)), 1),
            "success_rate": round(float(np.mean(succ)), 3)}


def record(model, run, path, seed=7):
    import imageio.v2 as imageio
    env_name = run_config(run)["env"]
    raw = build_env(env_name, seed, render_mode="rgb_array")
    vn = make_norm(run, {})
    raw_obs, _ = raw.reset(seed=seed)
    frames = []
    for _ in range(raw.max_steps):
        a, _ = model.predict(vn.normalize_obs(raw_obs[None]), deterministic=True)
        raw_obs, _, te, tr, info = raw.step(a[0])
        frames.append(raw.render())
        if te or tr:
            break
    for _ in range(25):                     # hold the last frame for half a second
        frames.append(frames[-1])
    imageio.mimsave(path, frames, fps=50)
    raw.close()
    print(f"[video] {len(frames)} frames -> {path} (solved={info.get('is_success', 'n/a')})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--robustness", action="store_true")
    ap.add_argument("--video", default=None, help="e.g. out.mp4 (needs imageio + imageio-ffmpeg)")
    ap.add_argument("--last", action="store_true", help="use final model instead of best")
    args = ap.parse_args()

    cfg = run_config(args.run)
    model = load(args.run, best=not args.last)
    tests = dict(TESTS) if args.robustness else {"clean": {}}
    if args.robustness and cfg["env"] == "maze_fixed3":
        tests["random_mazes"] = {"fixed_maze": False}

    res_path = os.path.join(args.run, "eval_results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    if args.episodes > 1 or not os.path.exists(res_path):
        print(f"{cfg['env']} / {cfg['policy']} / {cfg.get('wiring', '-')}")
        for name, kw in tests.items():
            r = run_episodes(model, args.run, kw, args.episodes)
            results[name] = r
            print(f"  {name:13s} success {r['success_rate'] * 100:5.1f}%   return {r['return']:8.1f}   "
                  f"length {r['length']:7.1f}")
        with open(res_path, "w") as f:
            json.dump(results, f, indent=2)
    if args.video:
        record(model, args.run, args.video)


if __name__ == "__main__":
    main()
