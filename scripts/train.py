"""Train a fly-brain (connectome-wired) controller with PPO.

Environments (--env): tray, maze_fixed3, maze3, maze3h, maze4, maze5 and the camera
versions *_vision (see ctb/envs/__init__.py)

Examples:
    # Mac: safe, slow, short (stops by itself after 30 min of compute)
    python scripts/train.py --policy connectome --profile mac

    # Tilt-maze puzzle, one fixed maze, then random mazes
    python scripts/train.py --env maze_fixed3 --policy connectome --profile kaggle
    python scripts/train.py --env maze3 --policy connectome --profile kaggle

    # Fly-robot brain (camera into the eye, commands out of descending neurons)
    python scripts/train.py --env tray_vision --subset data/flybrain_cam16_N2500.npz --profile kaggle

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
    ap.add_argument("--policy", choices=["connectome"], default="connectome")  # kept for older commands
    ap.add_argument("--wiring", choices=["real", "shuffled"], default="real")
    ap.add_argument("--subset", default="data/subset_flywire783_N1000.npz")
    ap.add_argument("--profile", choices=list(PROFILES), default="mac")
    ap.add_argument("--k_steps", type=int, default=None,
                    help="recurrent iterations per control step (default 4; 8 for the fly-robot brain)")
    ap.add_argument("--leak", type=float, default=None, help="neuron update rate (default 0.5; 1.0 fly-robot)")
    ap.add_argument("--gain", type=float, default=None, help="initial connection strength (default 1.5; 3.0 fly-robot)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--push_prob", type=float, default=0.0, help="random pushes on the ball (harder task)")
    ap.add_argument("--action_rate", type=float, default=0.0,
                    help="slow, smooth motors: max change of each joint target per step (action units, "
                         "range -1..1). 0 = off. 0.1 means a full swing takes 20 steps (0.4 s).")
    ap.add_argument("--random_start", type=float, default=0.0,
                    help="maze curriculum: fraction of training episodes that start the ball at a random "
                         "cell along the route (evaluation always uses the real start). Try 0.5.")
    ap.add_argument("--init_from", default=None,
                    help="fine-tune: start a NEW run from another run's best brain (run folder, or a run name "
                         "inside --runs_dir). Only used when this run has no checkpoint yet.")
    ap.add_argument("--lr", type=float, default=None, help="learning rate (default 3e-4; fine-tuning: 1e-4)")
    ap.add_argument("--std", type=float, default=None,
                    help="fine-tune: reset the exploration noise (action std) of the loaded brain, e.g. 0.25")
    ap.add_argument("--grasp_start", type=float, default=0.0,
                    help="panda_lift: share of TRAINING episodes that start with the gripper around the block "
                         "(half open, half already holding it)")
    ap.add_argument("--lift_reward", choices=["height", "goal"], default=None,
                    help="panda_lift task definition: 'height' (lift progress by height) or 'goal' (bring the "
                         "block to a point straight above its start; sideways carrying costs). Applies to "
                         "training AND evaluation, and is saved in the run's config.")
    ap.add_argument("--imitate", type=int, default=0,
                    help="panda_lift: first train the fly brain by imitating the scripted teacher (DAgger) for "
                         "this many rounds, then continue with PPO (needs --init_from, --privileged)")
    ap.add_argument("--imitate_steps", type=int, default=15_000, help="arm steps collected per later imitation round "
                    "(the first round collects 30,000 teacher steps)")
    ap.add_argument("--imitate_first_steps", type=int, default=30_000, help="teacher steps in the first imitation round")
    ap.add_argument("--imitate_epochs", type=float, default=6.0, help="passes over the imitation data per later round "
                    "(at most 2,000 updates; the first round makes 15 passes)")
    ap.add_argument("--privileged", action="store_true", default=None,
                    help="give the training-only critic the true simulator state (asymmetric actor-critic); "
                         "the fly brain never receives it")
    ap.add_argument("--grip_mode", choices=["delta", "absolute"], default=None,
                    help="panda: 'absolute' = the grip command IS the finger target (closed stays closed)")
    ap.add_argument("--set_std", type=float, default=None,
                    help="resumed run: reset the exploration noise (action std) to this value, ONCE "
                         "(remembered in the run folder, so a later restart does not reset it again)")
    ap.add_argument("--set_lr", type=float, default=None,
                    help="resumed run: change the learning rate from now on (e.g. 1e-4)")
    ap.add_argument("--edge_bias", type=float, default=0.0,
                    help="panda: share of TRAINING episodes with the block near the table edges / corners")
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
    from ctb.policies import ConnectomeCore, ConnectomePolicy
    from ctb.safety import SafetyCallback

    name = args.name or f"{args.env}-{args.policy}" + (f"-{args.wiring}" if args.policy == "connectome" else "") + f"-s{args.seed}"
    out = os.path.join(args.runs_dir, name)
    os.makedirs(out, exist_ok=True)
    model_path, vn_path = os.path.join(out, "model.zip"), os.path.join(out, "vecnormalize.pkl")

    env_kw = dict(push_prob=args.push_prob)
    if args.env.startswith(("drone", "panda")):
        import numpy as np
        brain = np.load(args.subset) if os.path.exists(args.subset) else None
        if brain is None or "facets" not in brain.files:
            raise SystemExit("drone/panda envs need a compound-eye brain: "
                             "python scripts/build_flybrain.py --eye compound --facets 160 --n 4000 --hops 5")
        env_kw["n_facets"] = int(brain["facets"])     # eye size must match the brain's eye
    if args.resume and os.path.exists(os.path.join(out, "config.json")):
        # keep the task / robot this run was last trained with, even if the job line leaves an option out
        _saved = json.load(open(os.path.join(out, "config.json")))
        for _k in ("lift_reward", "grip_mode", "privileged"):
            if getattr(args, _k) is None and _saved.get(_k) is not None:
                setattr(args, _k, _saved[_k])
    if args.grip_mode:            # robot property: applies to training and evaluation
        env_kw["grip_mode"] = args.grip_mode
        print(f"[train] gripper command: {args.grip_mode}")
    if args.privileged:           # critic input (training only); the fly brain never sees it
        env_kw["privileged"] = True
        print("[train] critic gets the true simulator state (asymmetric actor-critic); the fly brain does not")
    if args.lift_reward:          # task definition: applies to training and evaluation
        env_kw["lift_reward"] = args.lift_reward
        print(f"[train] lift reward: {args.lift_reward}")
    if args.action_rate > 0:      # robot property: applies to training and evaluation
        env_kw["action_rate"] = args.action_rate
        print(f"[train] slow motors: joint targets change by at most {args.action_rate} per step")
    Vec = SubprocVecEnv if cfg["vec"] == "subproc" else DummyVecEnv
    if cfg["vec"] == "subproc":
        # Each simulation process: ONE maths thread. Its matrices are tiny (arm IK, eye), and
        # several idle-spinning BLAS threads per process only compete for the CPU. The main
        # process already set its own thread count above, so this only affects the children.
        for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
            os.environ[var] = "1"
    train_kw = dict(env_kw, random_start=args.random_start) if args.random_start > 0 else dict(env_kw)
    if args.edge_bias > 0:                  # training only; evaluation keeps uniform block positions
        train_kw["edge_bias"] = args.edge_bias
        print(f"[train] {args.edge_bias:.0%} of training blocks placed near the table edges / corners")
    if args.grasp_start > 0:                # training only; evaluation always starts from scratch
        train_kw["grasp_start"] = args.grasp_start
        print(f"[train] {args.grasp_start:.0%} of training episodes start at the block (half open, half holding it)")
    if args.random_start > 0:
        print(f"[train] curriculum: {args.random_start:.0%} of training episodes start near the goal / at a random point on the route")
    venv = Vec([make_env(args.env, seed=args.seed * 1000 + i, **train_kw) for i in range(cfg["n_envs"])])
    venv = VecMonitor(venv)
    eval_env = VecMonitor(DummyVecEnv([make_env(args.env, seed=10_000 + args.seed, **env_kw)]))

    lr = args.lr or 3e-4
    init = None
    if args.init_from and not os.path.exists(model_path):
        init = args.init_from if os.path.isdir(args.init_from) else os.path.join(args.runs_dir, args.init_from)
        if not os.path.exists(os.path.join(init, "best_model.zip")):
            raise SystemExit(f"--init_from: no best_model.zip in {init}")

    do_imitate = False
    if init and args.imitate:
        # teacher -> student: a new run whose policy has the asymmetric critic. The fly brain's
        # learned weights are copied from the parent; imitation (DAgger) then PPO follow below.
        if not args.privileged:
            raise SystemExit("--imitate needs --privileged (the critic for the PPO phase)")
        import shutil
        from ctb.envs.panda import N_PRIV
        from ctb.policies import load_ppo
        shutil.copy(os.path.join(init, "subset.npz"), os.path.join(out, "subset.npz"))
        parent = json.load(open(os.path.join(init, "config.json")))
        # brain body-sense layout (last 13 of the brain inputs): hand position (3), velocity (3), finger
        # opening, gripper-closed flag, touch, previous command (4). Hide the flag and the previous
        # command from the student: copying them is the "copycat" shortcut of imitation learning.
        n_brain = venv.observation_space.shape[0] - N_PRIV
        mask = [n_brain - 13 + 7] + list(range(n_brain - 4, n_brain))
        parent.update(mask_inputs=mask)
        parent.update(lift_reward=args.lift_reward or "height", grip_mode=args.grip_mode or "delta", privileged=True,
                      n_priv=N_PRIV, env=args.env, name=name, init_from=os.path.abspath(init), lr=lr, std=args.std,
                      edge_bias=args.edge_bias, random_start=args.random_start, eval_episodes=cfg["eval_episodes"],
                      imitation={"rounds": args.imitate, "steps_per_round": args.imitate_steps,
                                 "teacher": "ctb.envs.panda.lift_teacher (true block position; training targets only)"})
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump(parent, f, indent=2)
        norm = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        policy_kwargs = dict(subset_path=os.path.abspath(os.path.join(out, "subset.npz")), k_steps=parent["k_steps"],
                             leak=parent["leak"], gain=parent["gain"], wiring=parent.get("wiring", "real"),
                             log_std_init=-0.5, n_priv=N_PRIV, mask_inputs=mask)
        model = PPO(ConnectomePolicy, norm, policy_kwargs=policy_kwargs, seed=args.seed, device=args.device,
                    n_steps=cfg["n_steps"], batch_size=cfg["batch_size"], n_epochs=cfg["n_epochs"],
                    learning_rate=lr, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                    ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, verbose=0)
        src = load_ppo(init, os.path.join(init, "best_model.zip"), device=args.device).policy.state_dict()
        dst = model.policy.state_dict()
        same = [k for k in dst if k in src and src[k].shape == dst[k].shape
                and not k.startswith(("mlp_extractor.critic", "value_net"))]
        for k in same:
            dst[k] = src[k]
        model.policy.load_state_dict(dst)
        venv = norm
        eval_env = VecNormalize(eval_env, training=False, norm_reward=False)
        print(f"[train] teacher -> student: fly brain copied from {init}/best_model.zip ({len(same)} tensors); "
              f"{args.imitate} imitation rounds of {args.imitate_steps:,} steps, then PPO (lr {lr})")
        do_imitate = True
        reset = False
    elif init:
        # fine-tune: copy the parent's brain wiring + settings, load its BEST brain, fresh step count
        import shutil
        from ctb.policies import load_ppo
        shutil.copy(os.path.join(init, "subset.npz"), os.path.join(out, "subset.npz"))
        parent = json.load(open(os.path.join(init, "config.json")))
        parent.update(lift_reward=args.lift_reward or "height", env=args.env, name=name, init_from=os.path.abspath(init), lr=lr, std=args.std, edge_bias=args.edge_bias,
                      random_start=args.random_start, eval_episodes=cfg["eval_episodes"])
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump(parent, f, indent=2)
        norm = VecNormalize.load(os.path.join(init, "vecnormalize.pkl"), venv)
        # inputs that never changed in the parent task (e.g. touch, unused in reach) have
        # ~zero variance there; reset them so the new signal is not blown up to the clip limit
        flat = norm.obs_rms.var < 1e-6
        if flat.any():
            norm.obs_rms.mean[flat], norm.obs_rms.var[flat] = 0.0, 1.0
            print(f"[train] reset normalisation of {int(flat.sum())} input(s) unused in the parent task")
        model = load_ppo(out, os.path.join(init, "best_model.zip"), env=norm, device=args.device,
                         custom={"learning_rate": lr, "lr_schedule": (lambda _: lr)})
        if args.std:
            with torch.no_grad():
                model.policy.log_std.fill_(float(np.log(args.std)))
        venv = norm
        eval_env = VecNormalize(eval_env, training=False, norm_reward=False)
        print(f"[train] fine-tuning from {init}/best_model.zip (was at {model.num_timesteps:,} steps): "
              f"lr={lr}" + (f", action std reset to {args.std}" if args.std else ""))
        model.num_timesteps = 0
        reset = True                       # this run counts its own steps from 0
    elif args.resume and os.path.exists(model_path):
        from ctb.policies import load_ppo
        try:
            norm = VecNormalize.load(vn_path, venv)
            model = load_ppo(out, model_path, env=norm, device=args.device)
        except Exception as e:                       # e.g. power cut during a save
            print(f"[train] latest checkpoint unreadable ({e}); using the previous one (.bak)")
            norm = VecNormalize.load(vn_path + ".bak", venv)
            model = load_ppo(out, model_path + ".bak", env=norm, device=args.device)
            model_path_used = model_path + ".bak"
        venv = norm
        eval_env = VecNormalize(eval_env, training=False, norm_reward=False)
        print(f"[train] resumed from {locals().get('model_path_used', model_path)} at {model.num_timesteps} steps")
        reset = False
        if args.lift_reward:
            cfg_path = os.path.join(out, "config.json")
            run_cfg = json.load(open(cfg_path))
            old_task = run_cfg.get("lift_reward") or "height"
            if old_task != args.lift_reward:
                # new task definition: scores before and after are not comparable, so the old
                # evaluation history and best brain are kept under other names and start fresh
                tag = f"_{old_task}_until_{model.num_timesteps // 1000}k"
                for f in ("evaluations.npz", "best_model.zip"):
                    src = os.path.join(out, f)
                    if os.path.exists(src):
                        os.replace(src, os.path.join(out, f.replace(".", tag + ".", 1)))
                run_cfg["lift_reward"] = args.lift_reward
                run_cfg.setdefault("task_changes", []).append({"steps": int(model.num_timesteps), "lift_reward": args.lift_reward})
                with open(cfg_path + ".tmp", "w") as fh:
                    json.dump(run_cfg, fh, indent=2)
                os.replace(cfg_path + ".tmp", cfg_path)
                print(f"[train] task changed: lift reward {old_task} -> {args.lift_reward} at {model.num_timesteps:,} steps; "
                      f"old evaluation history and best brain kept as *{tag}*, new ones start now")
        if args.set_lr:
            from stable_baselines3.common.utils import get_schedule_fn
            model.learning_rate = args.set_lr
            model.lr_schedule = get_schedule_fn(args.set_lr)
            print(f"[train] learning rate set to {args.set_lr}")
        if args.set_std:
            mark = os.path.join(out, f"std_reset_{args.set_std:g}.done")
            if os.path.exists(mark):
                print(f"[train] action std reset to {args.set_std:g} was already done earlier; keeping the learned value")
            else:
                with torch.no_grad():
                    model.policy.log_std.fill_(float(np.log(args.set_std)))
                open(mark, "w").write(str(model.num_timesteps))
                print(f"[train] action std reset to {args.set_std:g} (once) at {model.num_timesteps:,} steps")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_env = VecNormalize(eval_env, training=False, norm_reward=False)
        obs_dim = venv.observation_space.shape[0]
        act_dim = venv.action_space.shape[0]

        # The fly-robot brain is long (eye -> optic lobe -> central brain -> commands, 3-5
        # hops) and very sparse, so it needs more rounds and stronger connections for the
        # camera signal to reach the command neurons.
        flyrobot = os.path.exists(args.subset) and "vis_idx" in np.load(args.subset).files
        dk, dl, dg = (8, 1.0, 3.0) if flyrobot else (4, 0.5, 1.5)
        args.k_steps = args.k_steps or dk
        args.leak = args.leak if args.leak is not None else dl
        args.gain = args.gain if args.gain is not None else dg
        if flyrobot:
            print(f"[train] fly-robot brain: k_steps={args.k_steps} leak={args.leak} gain={args.gain}")

        if not os.path.exists(args.subset):
            raise SystemExit(f"{args.subset} not found. Run download_connectome.py and build_subset.py / build_flybrain.py first.")
        policy = ConnectomePolicy
        # keep a copy of the wiring next to the model, so the run loads on any machine
        import shutil
        shutil.copy(args.subset, os.path.join(out, "subset.npz"))
        policy_kwargs = dict(subset_path=os.path.abspath(os.path.join(out, "subset.npz")), k_steps=args.k_steps,
                             leak=args.leak, gain=args.gain, wiring=args.wiring, log_std_init=-0.5)
        core = ConnectomeCore(obs_dim, args.subset, args.k_steps, args.leak, args.gain, wiring=args.wiring)
        actor_params = sum(p.numel() for p in core.parameters()) + 2 * core.out_dim + core.out_dim * act_dim + act_dim
        del core

        model = PPO(
            policy, venv, policy_kwargs=policy_kwargs, seed=args.seed, device=args.device,
            n_steps=cfg["n_steps"], batch_size=cfg["batch_size"], n_epochs=cfg["n_epochs"],
            learning_rate=lr, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, verbose=0,
        )
        reset = True
        cfg_dump = dict(vars(args), **cfg, actor_params=int(actor_params), obs_dim=obs_dim, act_dim=act_dim)
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump(cfg_dump, f, indent=2)
        print(f"[train] {name}: actor params = {actor_params:,}")

    model.set_logger(configure(out, ["stdout", "csv"]))

    def save():
        """Power-cut safe: write to temporary files, keep the previous checkpoint as .bak,
        then swap the new files in (a rename is atomic, so a checkpoint is never half-written)."""
        import shutil

        def flush(path):                      # force the file onto the disk (survives a freeze)
            with open(path, "rb+") as fh:
                os.fsync(fh.fileno())
        tmp_model, tmp_vn = os.path.join(out, "model_tmp.zip"), vn_path + ".tmp"
        model.save(tmp_model)
        venv.save(tmp_vn)
        flush(tmp_model)
        flush(tmp_vn)
        for f in (model_path, vn_path):
            if os.path.exists(f) and os.path.getsize(f) > 0:      # never back up an empty file
                shutil.copy2(f, f + ".bak")
                flush(f + ".bak")
        os.replace(tmp_model, model_path)
        os.replace(tmp_vn, vn_path)
        dfd = os.open(out, os.O_RDONLY)       # make the renames themselves durable
        os.fsync(dfd)
        os.close(dfd)

    if do_imitate:
        from ctb.imitate import run_dagger
        run_dagger(model, venv, eval_env, out, init, iters=args.imitate, steps=args.imitate_steps,
                   eval_episodes=cfg["eval_episodes"], seed=args.seed, final_std=args.std or 0.15,
                   epochs=args.imitate_epochs, first_steps=args.imitate_first_steps,
                   log=lambda m: print(m, flush=True), save_fn=save)
        if model.num_timesteps >= cfg["total_steps"]:
            print(f"[train] finished after imitation. Saved to {out}.")
            venv.close()
            eval_env.close()
            return

    eval_cb = EvalCallback(eval_env, n_eval_episodes=cfg["eval_episodes"],
                           eval_freq=max(cfg["eval_every"] // cfg["n_envs"], 1),
                           log_path=out, best_model_save_path=out, deterministic=True, verbose=1)
    ev_file = os.path.join(out, "evaluations.npz")
    old = None
    if not reset and os.path.exists(ev_file):
        try:
            old = np.load(ev_file)
            if old["results"].ndim != 2 or old["results"].shape[1] != cfg["eval_episodes"]:
                print("[train] evaluation history used a different number of episodes; starting it fresh")
                old = None
        except Exception as e:                       # unreadable (e.g. power cut during a save)
            print(f"[train] evaluation history unreadable ({e}); starting it fresh")
            old = None
    if old is not None:
        # resumed run: keep the earlier evaluation history (otherwise the file restarts empty)
        keep = old["timesteps"] <= model.num_timesteps
        eval_cb.evaluations_timesteps = list(old["timesteps"][keep])
        eval_cb.evaluations_results = [list(r) for r in old["results"][keep]]
        eval_cb.evaluations_length = [list(r) for r in old["ep_lengths"][keep]]
        if "successes" in old.files and len(old["successes"]):
            eval_cb.evaluations_successes = [list(r) for r in old["successes"][keep]]
        # remember the best score so far: otherwise the first evaluation after a restart counts as
        # a "new best" and overwrites best_model.zip even when it is worse
        if keep.any():
            eval_cb.best_mean_reward = float(old["results"][keep].mean(axis=1).max())
            print(f"[train] best evaluation so far: {eval_cb.best_mean_reward:.1f} (best_model.zip kept until beaten)")
    callbacks = [eval_cb]
    # Save model + normalisation every 50k steps, so a crashed or timed-out session
    # (Kaggle limit, laptop sleep) can continue with --resume instead of starting over.
    from stable_baselines3.common.callbacks import EveryNTimesteps
    from stable_baselines3.common.callbacks import BaseCallback as _Base

    class _Save(_Base):
        def _on_step(self):
            save()
            return True
    callbacks.append(EveryNTimesteps(n_steps=50_000, callback=_Save()))

    # One compact line per 50k steps about the TRAINING episodes (noisy policy, and with
    # --random_start including the easier starts). Shows progress before the eval does.
    class _Progress(_Base):
        def _on_step(self):
            m = self.model
            rews = [e["r"] for e in m.ep_info_buffer]
            succ = list(m.ep_success_buffer)
            line = f"[progress] {m.num_timesteps:,} steps | training episodes: return {np.mean(rews):.2f}" if rews else \
                   f"[progress] {m.num_timesteps:,} steps"
            if succ:
                line += f" | solved {100 * np.mean(succ):.0f}%"
            print(line, flush=True)
            return True
    callbacks.append(EveryNTimesteps(n_steps=50_000, callback=_Progress()))

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
