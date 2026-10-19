"""Teacher -> student imitation (DAgger) for the fly-brain Panda.

The TEACHER is a hand-written controller that knows the true block position (ctb.envs.panda.
lift_teacher). The STUDENT is the fly-brain network, which only gets its two compound eyes and
its body sense. Each round:

  1. run the arm for `steps` steps; at every step the teacher says what it would do here.
     Who actually moves the arm: the teacher with probability beta (1.0 in round 0, halving
     every round), otherwise the student, so later rounds visit the situations the STUDENT gets
     itself into and the teacher shows how to get out of them (DAgger, Ross et al. 2011);
  2. add (what the student saw, what the teacher did) to the data set;
  3. train the fly brain's connection strengths to output the teacher's moves (supervised);
  4. test the student alone on fresh layouts (normal starts, no teacher), exactly like the
     evaluation chart of a PPO run, and keep the best brain.

The teacher's knowledge only shapes the training targets. The fly brain never receives the
block position as an input (the policy slices it away), and the teacher is never used in
evaluation, videos or `fly eval`.

Output is an ordinary run folder (model.zip, vecnormalize.pkl, evaluations.npz, best_model.zip),
so PPO can continue from it and the dashboard, videos and `fly eval` work as usual. Each round is
checkpointed, so a power cut costs at most one round.
"""
from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np
import torch

from ctb.envs.panda import N_PRIV


def _actor_params(policy):
    ext = policy.mlp_extractor
    return list(ext.core.parameters()) + list(ext.readout_norm.parameters()) + list(policy.action_net.parameters())


def _actor_mean(policy, obs_brain):
    """Fly-brain action (before clipping) from the brain's own inputs only."""
    ext = policy.mlp_extractor
    if len(ext.mask_idx):                          # same input mask as the policy itself
        obs_brain = obs_brain.clone()
        obs_brain[:, ext.mask_idx] = 0.0
    return policy.action_net(ext.readout_norm(ext.core(obs_brain)))


def _save_stats(norm, path):
    with open(path, "wb") as f:
        pickle.dump({"mean": norm.obs_rms.mean, "var": norm.obs_rms.var, "count": norm.obs_rms.count}, f)


def _load_stats(norm, path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    norm.obs_rms.mean[:], norm.obs_rms.var[:], norm.obs_rms.count = d["mean"], d["var"], d["count"]


def run_dagger(model, norm, eval_norm, out, parent_dir, iters=10, steps=15_000, cap=120_000, epochs=6,
               batch=256, lr=1e-3, eval_episodes=50, seed=0, log=print, final_std=0.15, save_fn=None,
               first_steps=30_000, first_epochs=15, max_updates=2000, teacher_noise=0.15):
    """model: PPO whose policy has n_priv = N_PRIV; norm / eval_norm: VecNormalize around the
    training / evaluation envs (privileged obs). Writes the run folder and returns the model."""
    venv = norm.venv
    n_envs = venv.num_envs
    ext = model.policy.mlp_extractor
    nb = ext.n_brain
    rng = np.random.default_rng(seed)
    ckpt = os.path.join(out, "imitate_state.json")
    ev_file = os.path.join(out, "evaluations.npz")
    state = json.load(open(ckpt)) if os.path.exists(ckpt) else {"round": 0, "steps": 0, "best": -1e9}
    hist = dict(np.load(ev_file)) if os.path.exists(ev_file) and state["round"] > 0 else \
        {"timesteps": np.zeros(0, int), "results": np.zeros((0, eval_episodes)),
         "ep_lengths": np.zeros((0, eval_episodes), int), "successes": np.zeros((0, eval_episodes))}

    if state["round"] > 0:                         # resumed after a stop / power cut
        model.policy.load_state_dict(torch.load(os.path.join(out, "imitate_policy.pt")))
        _load_stats(norm, os.path.join(out, "imitate_obs_stats.pkl"))
        log(f"[imitate] resumed at round {state['round']} ({state['steps']:,} steps)")
    else:
        # brain inputs: keep the parent's normalisation (the brain was trained with it)
        with open(os.path.join(parent_dir, "vecnormalize.pkl"), "rb") as f:
            parent = pickle.load(f)
        m = len(parent.obs_rms.mean)              # parent may itself have had privileged inputs after the brain's
        assert m >= nb, f"parent brain has {m} inputs, this one {nb}"
        norm.obs_rms.mean[:nb], norm.obs_rms.var[:nb] = parent.obs_rms.mean[:nb], parent.obs_rms.var[:nb]
        norm.obs_rms.count = parent.obs_rms.count
        norm.ret_rms = parent.ret_rms
    norm.training = False                          # frozen while imitating (the student's inputs stay fixed)
    eval_norm.obs_rms = norm.obs_rms

    opt = torch.optim.Adam(_actor_params(model.policy), lr=lr)
    X = np.zeros((0, nb), np.float16)
    Y = np.zeros((0, 4), np.float32)
    data_f = os.path.join(out, "imitate_data.npz")
    if state["round"] > 0 and os.path.exists(data_f):      # the data collected so far (DAgger aggregates it)
        with np.load(data_f) as d:
            X, Y = d["X"], d["Y"]
        log(f"[imitate] data set restored: {len(X):,} examples")

    def evaluate():
        env = eval_norm.venv
        rets, lens, succ = [], [], []
        obs = env.reset()
        cur_r, cur_l = 0.0, 0
        while len(rets) < eval_episodes:
            o = torch.as_tensor(eval_norm.normalize_obs(obs)[:, :nb], dtype=torch.float32)
            with torch.no_grad():
                a = _actor_mean(model.policy, o).clamp(-1, 1).numpy()
            obs, r, done, info = env.step(a)
            cur_r += float(r[0]); cur_l += 1
            if done[0]:
                rets.append(cur_r); lens.append(cur_l); succ.append(float(info[0].get("is_success", False)))
                cur_r, cur_l = 0.0, 0
        return np.array(rets), np.array(lens), np.array(succ)

    while state["round"] < iters:
        k = state["round"]
        beta = 0.5 ** k                            # who drives: teacher 100 %, 50 %, 25 % ...
        n_steps = first_steps if k == 0 else steps
        t0 = time.time()
        log(f"[imitate] round {k + 1}/{iters}: collecting {n_steps:,} steps (teacher drives {beta:.0%}) ...")
        # ---- 1. collect: teacher labels everywhere, mixed control ------------------------------
        obs = venv.reset()
        xs, ys, ep_r, ep_s = [], [], [], []
        cur = np.zeros(n_envs)
        priv_raw = []
        for _ in range(n_steps // n_envs):
            teacher = np.array(venv.env_method("teacher_action"), dtype=np.float32)
            nobs = norm.normalize_obs(obs)
            if k == 0:
                priv_raw.append(obs[:, nb:].copy())
            with torch.no_grad():
                student = _actor_mean(model.policy, torch.as_tensor(nobs[:, :nb], dtype=torch.float32)).clamp(-1, 1).numpy()
            use_teacher = rng.random(n_envs) < beta
            # a little noise on whoever drives, so the data covers slightly-off situations too
            drive = np.where(use_teacher[:, None], teacher, student)
            act = np.clip(drive + rng.normal(0, teacher_noise, drive.shape), -1, 1)
            xs.append(nobs[:, :nb].astype(np.float16))
            ys.append(teacher)
            obs, r, done, info = venv.step(act)
            cur += r
            for i in np.where(done)[0]:
                ep_r.append(cur[i]); ep_s.append(float(info[i].get("is_success", False))); cur[i] = 0
        if k == 0:                                 # critic inputs: statistics of the true state
            pr = np.concatenate(priv_raw)
            norm.obs_rms.mean[nb:], norm.obs_rms.var[nb:] = pr.mean(0), pr.var(0) + 1e-6
        state["steps"] += n_steps
        X = np.concatenate([X] + xs)
        Y = np.concatenate([Y] + ys)
        if len(X) > cap:                           # keep a random subset of the old data + all new data
            new = n_steps
            keep_old = rng.choice(len(X) - new, cap - new, replace=False)
            idx = np.concatenate([np.sort(keep_old), np.arange(len(X) - new, len(X))])
            X, Y = X[idx], Y[idx]
        t_collect = time.time() - t0
        # ---- 2. train the fly brain on the teacher's moves --------------------------------------
        log(f"[imitate] round {k + 1}/{iters}: collected in {t_collect:.0f}s; learning from {len(X):,} examples ...")
        model.policy.train()
        # round 0 learns the teacher thoroughly; later rounds a fixed, sufficient amount
        n_up = int(first_epochs * len(X) / batch) if k == 0 else min(max_updates, int(epochs * len(X) / batch))
        n_up = max(1, n_up)
        losses = []
        for u in range(n_up):
            b = rng.integers(0, len(X), batch)
            pred = _actor_mean(model.policy, torch.as_tensor(X[b], dtype=torch.float32))
            loss = ((pred - torch.as_tensor(Y[b])) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(_actor_params(model.policy), 1.0)
            opt.step()
            losses.append(loss.item())
            if (u + 1) % max(1, n_up // 4) == 0 and u + 1 < n_up:
                log(f"[imitate] round {k + 1}/{iters}: learning {100 * (u + 1) // n_up}% "
                    f"(imitation loss {np.mean(losses[-50:]):.4f})")
        model.policy.eval()
        t_train = time.time() - t0 - t_collect
        # ---- 3. test the student alone ----------------------------------------------------------
        log(f"[imitate] round {k + 1}/{iters}: testing the student alone on {eval_episodes} episodes ...")
        rets, lens, succ = evaluate()
        hist["timesteps"] = np.append(hist["timesteps"], state["steps"])
        for key, v in (("results", rets), ("ep_lengths", lens), ("successes", succ)):
            hist[key] = np.concatenate([hist[key], v[None]])
        np.savez(ev_file, **hist)
        if rets.mean() > state["best"]:
            state["best"] = float(rets.mean())
            state["best_round"] = k + 1
            model.save(os.path.join(out, "best_model.zip"))
            torch.save(model.policy.state_dict(), os.path.join(out, "imitate_best_policy.pt"))
        log(f"[progress] {state['steps']:,} steps | training episodes: return {np.mean(ep_r) if ep_r else 0:.2f} | "
            f"solved {100 * np.mean(ep_s) if ep_s else 0:.0f}%")
        log(f"Eval num_timesteps={state['steps']}, episode_reward={rets.mean():.2f} +/- {rets.std():.2f}")
        log(f"Success rate: {100 * succ.mean():.2f}%")
        log(f"[imitate] round {k + 1}/{iters}: teacher drives {beta:.0%} | data {len(X):,} | imitation loss "
            f"{np.mean(losses[-50:]):.4f} | student alone: return {rets.mean():.1f}, solved {succ.mean():.0%} | "
            f"{t_collect:.0f}s collect, {t_train:.0f}s learn")
        # ---- 4. checkpoint ----------------------------------------------------------------------
        state["round"] = k + 1
        torch.save(model.policy.state_dict(), os.path.join(out, "imitate_policy.pt"))
        with open(data_f + ".tmp", "wb") as f:                # the aggregated data, so a restart keeps it
            np.savez(f, X=X, Y=Y)
        os.replace(data_f + ".tmp", data_f)
        _save_stats(norm, os.path.join(out, "imitate_obs_stats.pkl"))
        with open(ckpt + ".tmp", "w") as f:
            json.dump(state, f)
        os.replace(ckpt + ".tmp", ckpt)

    # hand over to PPO from the BEST round (later rounds can be worse: once the student drives
    # itself, the data fills with its own near-misses), modest exploration noise, normalisation learns again
    best_f = os.path.join(out, "imitate_best_policy.pt")
    if os.path.exists(best_f):
        model.policy.load_state_dict(torch.load(best_f))
        log(f"[imitate] PPO starts from the best round ({state.get('best_round', '?')}), not the last")
    with torch.no_grad():
        model.policy.log_std.fill_(float(np.log(final_std)))
    model.num_timesteps = state["steps"]
    norm.training = True
    if save_fn is not None:
        save_fn()
    log(f"[imitate] done: {state['steps']:,} steps, best student return {state['best']:.1f}; "
        f"handing over to PPO with action std {final_std}")
    return model
