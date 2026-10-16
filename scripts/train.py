"""Train a PPO agent with a connectome-shaped or MLP policy.

Environments (--env): tray, maze_fixed3, maze3, maze4, maze5 (see ctb/envs/__init__.py)

Examples:
    # Mac: safe, slow, short (stops by itself after 30 min of compute)
    python scripts/train.py --policy connectome --profile mac

    # Tilt-maze puzzle, one fixed maze, then random mazes
    python scripts/train.py --env maze_fixed3 --policy connectome --profile kaggle
    python scripts/train.py --env maze3 --policy connectome --profile kaggle

    # Same-size MLP baseline
    python scripts/train.py --policy mlp --profile mac

    # Shuffled-wiring control (same edges/weights, random connections)
    python scripts/train.py --policy connectome --wiring shuffled --profile kaggle

    # Continue a stopped run
    python scripts/train.py --policy connectome --profile mac --resume

Outputs go to runs/<env>-<policy>-<wiring>-s<seed>/ : model.zip, vecnormalize.pkl, config.json,
progress.csv (training log), evaluations.npz (eval curve).
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ctb.profiles import ENV_NAMES, PROFILES  # noqa: E402 (no heavy imports)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=ENV_NAMES, default="tray")
    ap.add_argument("--policy", choices=["connectome", "mlp"], default="connectome")
    ap.add_argument("--wiring", choices=["real", "shuffled"], default="real")
    ap.add_argument("--subset", default="data/subset_flywire783_N1000.npz")
    ap.add_argument("--profile", choices=list(PROFILES), default="mac")
    ap.add_argument("--k_steps", type=int, default=4, help="recurrent iterations per control step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--push_prob", type=float, default=0.0, help="random pushes on the ball (harder task)")
    # profile overrides
    for k, v in PROFILES["mac"].items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            ap.add_argument(f"--{k}", type=type(v), default=None)
    args = ap.parse_args()
    cfg = dict(PROFILES[args.profile])
    for k in cfg:
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    return args, cfg


def main():
    args, cfg = parse()

    # Thread limits must be set before numpy/torch spin up their pools.
    from ctb.safety import limit_threads, lower_priority
    limit_threads(cfg["threads"])
    if cfg["nice"]:
        lower_priority(cfg["nice"])

    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, EvalCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

    from ctb.envs import make_env
    from ctb.policies import ConnectomeCore, ConnectomePolicy, matched_mlp_width
    from ctb.safety import SafetyCallback

    name = args.name or f"{args.env}-{args.policy}" + (f"-{args.wiring}" if args.policy == "connectome" else "") + f"-s{args.seed}"
    out = os.path.join(args.runs_dir, name)
    os.makedirs(out, exist_ok=True)
    model_path, vn_path = os.path.join(out, "model.zip"), os.path.join(out, "vecnormalize.pkl")

    env_kw = dict(push_prob=args.push_prob)
    Vec = SubprocVecEnv if cfg["vec"] == "subproc" else DummyVecEnv
    venv = Vec([make_env(args.env, seed=args.seed * 1000 + i, **env_kw) for i in range(cfg["n_envs"])])
    venv = VecMonitor(venv)
    eval_env = VecMonitor(DummyVecEnv([make_env(args.env, seed=10_000 + args.seed, **env_kw)]))

    if args.resume and os.path.exists(model_path):
        venv = VecNormalize.load(vn_path, venv)
        eval_env = VecNormalize(eval_env, training=False, norm_reward=False)
        from ctb.policies import load_ppo
        model = load_ppo(out, model_path, env=venv, device=args.device)
        print(f"[train] resumed from {model_path} at {model.num_timesteps} steps")
        reset = False
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_env = VecNormalize(eval_env, training=False, norm_reward=False)
        obs_dim = venv.observation_space.shape[0]
        act_dim = venv.action_space.shape[0]

        if args.policy == "connectome":
            if not os.path.exists(args.subset):
                raise SystemExit(f"{args.subset} not found. Run download_connectome.py and build_subset.py first.")
            policy = ConnectomePolicy
            # keep a copy of the wiring next to the model, so the run loads on any machine
            import shutil
            shutil.copy(args.subset, os.path.join(out, "subset.npz"))
            policy_kwargs = dict(subset_path=os.path.abspath(os.path.join(out, "subset.npz")), k_steps=args.k_steps,
                                 wiring=args.wiring, log_std_init=-0.5)
            core = ConnectomeCore(obs_dim, args.subset, args.k_steps, wiring=args.wiring)
            actor_params = sum(p.numel() for p in core.parameters()) + 2 * core.out_dim + core.out_dim * act_dim + act_dim
            del core
        else:
            # same number of actor parameters as the connectome policy
            core = ConnectomeCore(obs_dim, args.subset, args.k_steps) if os.path.exists(args.subset) else None
            target = (sum(p.numel() for p in core.parameters()) + 2 * core.out_dim + core.out_dim * act_dim + act_dim) if core else 56_000
            h = matched_mlp_width(obs_dim, act_dim, target)
            policy = "MlpPolicy"
            policy_kwargs = dict(net_arch=dict(pi=[h, h], vf=[64, 64]), activation_fn=torch.nn.Tanh,
                                 log_std_init=-0.5)
            actor_params = obs_dim * h + h + h * h + h + h * act_dim + act_dim

        model = PPO(
            policy, venv, policy_kwargs=policy_kwargs, seed=args.seed, device=args.device,
            n_steps=cfg["n_steps"], batch_size=cfg["batch_size"], n_epochs=cfg["n_epochs"],
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, verbose=0,
        )
        reset = True
        cfg_dump = dict(vars(args), **cfg, actor_params=int(actor_params), obs_dim=obs_dim, act_dim=act_dim)
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump(cfg_dump, f, indent=2)
        print(f"[train] {name}: actor params = {actor_params:,}")

    model.set_logger(configure(out, ["stdout", "csv"]))

    def save():
        model.save(model_path)
        venv.save(vn_path)

    callbacks = [EvalCallback(eval_env, n_eval_episodes=cfg["eval_episodes"],
                              eval_freq=max(cfg["eval_every"] // cfg["n_envs"], 1),
                              log_path=out, best_model_save_path=out, deterministic=True, verbose=1)]
    safety = None
    if cfg["safety"]:
        safety = SafetyCallback(save, work_s=cfg["work_s"], rest_s=cfg["rest_s"],
                                max_minutes=cfg["max_minutes"], min_free_gb=cfg["min_free_gb"])
        callbacks.append(safety)

    remaining = max(cfg["total_steps"] - model.num_timesteps, 0)
    print(f"[train] training for {remaining:,} steps -> {out}")
    try:
        model.learn(remaining, callback=CallbackList(callbacks), reset_num_timesteps=reset, progress_bar=False)
    except KeyboardInterrupt:
        print("\n[train] interrupted, saving...")
    save()
    venv.close()
    eval_env.close()
    why = safety.stop_reason if safety and safety.stop_reason else "finished"
    print(f"[train] {why}. Saved to {out} at {model.num_timesteps:,} steps. Use --resume to continue.")


if __name__ == "__main__":
    main()
