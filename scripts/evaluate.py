"""Evaluate a trained run, optionally under robustness tests, and record a video.

Examples:
    python scripts/evaluate.py --run runs/connectome-real-s0
    python scripts/evaluate.py --run runs/connectome-real-s0 --robustness
    python scripts/evaluate.py --run runs/connectome-real-s0 --video out.mp4

Robustness tests (none of these were seen in training):
    noise     Gaussian noise on every sensor
    blind     ball position/velocity hidden from the policy ("vision loss")
    heavy     ball 3x heavier
    push      random pushes on the ball
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
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from ctb.envs import TrayBalanceEnv  # noqa: E402
from ctb.policies import ConnectomePolicy  # noqa: E402

TESTS = {
    "clean": {},
    "noise_0.1": {"obs_noise": 0.1},
    "noise_0.3": {"obs_noise": 0.3},
    "blind": {"ball_blind": True},
    "heavy_ball": {"ball_mass_scale": 3.0},
    "push": {"push_prob": 0.05},
}


def load(run, best=True):
    path = os.path.join(run, "best_model.zip" if best and os.path.exists(os.path.join(run, "best_model.zip")) else "model.zip")
    model = PPO.load(path, device="cpu", custom_objects={"policy_class": ConnectomePolicy}
                     if "connectome" in json.load(open(os.path.join(run, "config.json")))["policy"] else None)
    return model


def make_norm(run, env_kw):
    venv = DummyVecEnv([lambda: TrayBalanceEnv(**env_kw)])
    vn = VecNormalize.load(os.path.join(run, "vecnormalize.pkl"), venv)
    vn.training = False
    vn.norm_reward = False
    return vn


def run_episodes(model, run, env_kw, episodes, seed=123):
    rets, lens = [], []
    for ep in range(episodes):
        vn = make_norm(run, dict(env_kw, seed=seed + ep))
        obs = vn.reset()
        done, ret, n = False, 0.0, 0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, d, _ = vn.step(a)
            ret += float(r[0]); n += 1; done = bool(d[0])
        rets.append(ret); lens.append(n)
        vn.close()
    return float(np.mean(rets)), float(np.mean(lens)), float(np.mean(np.array(lens) >= 500))


def record(model, run, path, seed=7, steps=500):
    import imageio.v2 as imageio
    raw = TrayBalanceEnv(seed=seed, render_mode="rgb_array")
    vn = make_norm(run, {})
    raw_obs, _ = raw.reset(seed=seed)
    frames = []
    for _ in range(steps):
        a, _ = model.predict(vn.normalize_obs(raw_obs[None]), deterministic=True)
        raw_obs, _, te, tr, _ = raw.step(a[0])
        frames.append(raw.render())
        if te or tr:
            break
    imageio.mimsave(path, frames, fps=50)
    raw.close()
    print(f"[video] {len(frames)} frames -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--robustness", action="store_true")
    ap.add_argument("--video", default=None, help="e.g. out.mp4 (needs imageio + imageio-ffmpeg)")
    ap.add_argument("--last", action="store_true", help="use final model instead of best")
    args = ap.parse_args()

    model = load(args.run, best=not args.last)
    tests = TESTS if args.robustness else {"clean": {}}
    results = {}
    for name, kw in tests.items():
        ret, ln, ok = run_episodes(model, args.run, kw, args.episodes)
        results[name] = {"return": round(ret, 1), "length": round(ln, 1), "full_episode_rate": ok}
        print(f"{name:12s} return {ret:8.1f}   length {ln:6.1f}/500   survived {ok * 100:5.1f}%")
    with open(os.path.join(args.run, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    if args.video:
        record(model, args.run, args.video)


if __name__ == "__main__":
    main()
