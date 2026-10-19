"""Evaluate a trained run, optionally under robustness tests, and record a video.

Examples:
    python scripts/evaluate.py --run runs/tray-connectome-real-s0
    python scripts/evaluate.py --run runs/maze3-connectome-real-s0 --robustness
    python scripts/evaluate.py --run runs/maze3-connectome-real-s0 --episodes 1 --video maze.mp4

Robustness tests (none of these were seen in training; drone runs have their own set,
see TESTS_DRONE):
    noise_0.1/0.3  Gaussian noise on every sensor
    blind          ball position/velocity hidden from the policy ("vision loss")
    heavy_ball     ball 3x heavier
    push           random pushes on the ball
    random_mazes   (only for runs trained on maze_fixed3 or maze_fixed3_vision) unseen mazes

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


# Drone (compound eye). Training already has light wind, gusts, eye noise and up to 2
# distractor balls, so "clean" means training conditions.
TESTS_DRONE = {
    "clean": {},
    "no_wind": {"wind": 0.0, "gusts": 0.0},
    "strong_wind": {"wind": 1.0, "gusts": 0.4},
    "eye_noise_x3": {"eye_noise": 0.03},
    "3_distractors": {"max_distractors": 3},
    "heavier_drone": {"mass_jitter": 0.12},
    "target_hidden": {"target_visible": False},
}


# Panda (two compound eyes). Training has eye noise, light changes and up to 2 decoys.
TESTS_PANDA = {
    "clean": {},
    "no_decoys": {"max_distractors": 0},
    "eye_noise_x3": {"eye_noise": 0.03},
    "dim_or_bright_x2": {"light_jitter": 0.4},          # brightness x 0.6 .. 1.4 (training default x 0.8 .. 1.2)
    "light_0.4x_or_2.5x": {"light_choices": (0.4, 2.5)},  # beyond even the wide-light training (x 0.5 .. 2)
    "target_hidden": {"target_visible": False},
}


def run_config(run):
    cfg = json.load(open(os.path.join(run, "config.json")))
    cfg.setdefault("env", "tray")          # runs from before the maze update
    return cfg


def load(run, best=True):
    best_path = os.path.join(run, "best_model.zip")
    path = best_path if best and os.path.exists(best_path) else os.path.join(run, "model.zip")
    return load_ppo(run, path)


def run_env_kw(run):
    """Robot settings a run was trained with (so evaluation uses the same robot)."""
    cfg = run_config(run)
    kw = {"action_rate": cfg["action_rate"]} if cfg.get("action_rate") else {}
    if cfg.get("lift_reward"):              # task definition (reward + success rule), not a training aid
        kw["lift_reward"] = cfg["lift_reward"]
    if cfg.get("grip_mode"):                # robot property
        kw["grip_mode"] = cfg["grip_mode"]
    if cfg.get("privileged"):               # obs layout the policy expects (critic part; the brain ignores it)
        kw["privileged"] = True
    if cfg["env"].startswith(("drone", "panda")):
        import numpy as np
        brain = os.path.join(run, "subset.npz")
        if os.path.exists(brain):
            kw["n_facets"] = int(np.load(brain)["facets"])
    return kw


def make_norm(run, env_kw, seed=0, best=False):
    """Input normalisation for a run. best=True: the statistics saved together with best_model.zip
    (best_vecnormalize.pkl), so the best brain sees its inputs scaled as when it was chosen; runs from
    before that file existed fall back to the latest statistics (vecnormalize.pkl)."""
    env_name = run_config(run)["env"]
    env_kw = {**run_env_kw(run), **env_kw}
    venv = DummyVecEnv([lambda: build_env(env_name, seed, **env_kw)])
    f = os.path.join(run, "best_vecnormalize.pkl")
    if not (best and os.path.exists(f)):
        f = os.path.join(run, "vecnormalize.pkl")
    vn = VecNormalize.load(f, venv)
    vn.training = False
    vn.norm_reward = False
    return vn


def run_episodes(model, run, env_kw, episodes, seed=123, best=False):
    rets, lens, succ = [], [], []
    for ep in range(episodes):
        vn = make_norm(run, env_kw, seed + ep, best=best)
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


def add_eye_view(frame, cam_img, size=144):
    """Paste what the brain sees (the 16x16 camera image) into the top-left corner."""
    img = np.flipud(cam_img)                         # row 0 = far side of the tray, on top
    rgb = np.zeros(img.shape + (3,))
    rgb[..., 0] = rgb[..., 1] = rgb[..., 2] = np.clip(img, 0, 1)      # ball/walls/goal: grey-white
    rgb[img < 0] = (0.8, 0.1, 0.1)                                    # holes: red
    k = size // img.shape[0]
    big = (np.kron(rgb, np.ones((k, k, 1))) * 255).astype(np.uint8)
    out = frame.copy()
    out[6:6 + big.shape[0], 6:6 + big.shape[1]] = big
    out[4:6, 4:8 + big.shape[1]] = out[6 + big.shape[0]:8 + big.shape[0], 4:8 + big.shape[1]] = 255
    out[4:8 + big.shape[0], 4:6] = out[4:8 + big.shape[0], 6 + big.shape[1]:8 + big.shape[1]] = 255
    return out


def record(model, run, path, seed=7, best=False):
    import imageio.v2 as imageio
    env_name = run_config(run)["env"]
    raw = build_env(env_name, seed, render_mode="rgb_array", **run_env_kw(run))
    vn = make_norm(run, {}, best=best)
    raw_obs, _ = raw.reset(seed=seed)
    frames = []
    for _ in range(raw.max_steps):
        a, _ = model.predict(vn.normalize_obs(raw_obs[None]), deterministic=True)
        raw_obs, _, te, tr, info = raw.step(a[0])
        frame = raw.render()
        if getattr(raw, "vision", False):
            frame = add_eye_view(frame, raw.camera_frame())
        frames.append(frame)
        if te or tr:
            break
    for _ in range(25):                     # hold the last frame for half a second
        frames.append(frames[-1])
    imageio.mimsave(path, frames, fps=raw.metadata.get("render_fps", 50))
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
    named = TESTS_DRONE if cfg["env"].startswith("drone") else TESTS_PANDA if cfg["env"].startswith("panda") else TESTS
    tests = dict(named) if args.robustness else {"clean": {}}
    if args.robustness and cfg["env"].startswith("maze_fixed3"):
        tests["random_mazes"] = {"fixed_maze": False}

    res_path = os.path.join(args.run, "eval_results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    if args.episodes > 1 or not os.path.exists(res_path):
        print(f"{cfg['env']} / {cfg['policy']} / {cfg.get('wiring', '-')}")
        for name, kw in tests.items():
            r = run_episodes(model, args.run, kw, args.episodes, best=not args.last)
            results[name] = r
            print(f"  {name:13s} success {r['success_rate'] * 100:5.1f}%   return {r['return']:8.1f}   "
                  f"length {r['length']:7.1f}")
        with open(res_path, "w") as f:
            json.dump(results, f, indent=2)
    if args.video:
        record(model, args.run, args.video, best=not args.last)


if __name__ == "__main__":
    main()
