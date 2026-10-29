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

With memory (--memory, ConnectomeRecurrentPolicy) the student's neuron activity carries over from
step to step, during collection, testing and learning: the data is kept as stretches of consecutive
steps together with the student's neuron state saved every 16 steps. The brain learns on 48-step
pieces, each started from the saved state and run through a 24-step warm-up (burn-in) first, with
the error flowing back through time (truncated backpropagation through time) -- the stored-state +
burn-in scheme of R2D2 (Kapturowski et al. 2019), which beats starting from zero.

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
    extra = [ext.mem_logit] if hasattr(ext, "mem_logit") else []          # memory: the carry-over gates
    dn = list(ext.readout_norm_dn.parameters()) if getattr(ext, "readout_norm_dn", None) is not None else []
    return list(ext.core.parameters()) + list(ext.readout_norm.parameters()) + dn + list(policy.action_net.parameters()) + extra


def _actor_mean(policy, obs_brain):
    """Fly-brain action (before clipping) from the brain's own inputs only."""
    ext = policy.mlp_extractor
    if len(ext.mask_idx):                          # same input mask as the policy itself
        obs_brain = obs_brain.clone()
        obs_brain[:, ext.mask_idx] = 0.0
    return policy.action_net(ext.readout(ext.core(obs_brain)))


def _actor_step(policy, obs_brain, h):
    """With memory: action from the brain's inputs and its neuron state h -> (action, new state)."""
    out, h = policy.mlp_extractor.step_actor(obs_brain, h)
    return policy.action_net(out), h


# memory: stored stretch, learned piece, warm-up, pieces per update, neuron state saved every H_EVERY steps
BLOCK, CHUNK, BURN, N_CHUNKS, H_EVERY = 256, 48, 24, 32, 16     # 32 pieces per update: ~1.5x faster per step learned


def _save_stats(norm, path):
    with open(path, "wb") as f:
        pickle.dump({"mean": norm.obs_rms.mean, "var": norm.obs_rms.var, "count": norm.obs_rms.count}, f)


def _load_stats(norm, path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    norm.obs_rms.mean[:], norm.obs_rms.var[:], norm.obs_rms.count = d["mean"], d["var"], d["count"]


def run_dagger(model, norm, eval_norm, out, parent_dir, iters=10, steps=15_000, cap=120_000, epochs=6,
               batch=256, lr=1e-3, eval_episodes=50, seed=0, log=print, final_std=0.15, save_fn=None,
               first_steps=30_000, first_epochs=15, max_updates=2000, teacher_noise=0.15, remeasure=None,
               grip_w=1.0, segment=0):
    """model: PPO whose policy has n_priv = N_PRIV; norm / eval_norm: VecNormalize around the
    training / evaluation envs (privileged obs). Writes the run folder and returns the model.
    grip_w: weight of the gripper error on the steps where the teacher decides to close an open hand (a rare,
    one-step event among many 'keep open' steps: with a plain mean error the student learned the average,
    0.18 instead of -1, and hovered at the tomato for the rest of the episode).
    segment > 0: who drives (teacher with probability beta, else the student) is decided per stretch of this many
    steps instead of per step, so the student's own mistakes run on and get labelled (as in HG-DAgger: the novice
    drives, the expert labels); per step, the teacher took over every few steps and the hover never happened."""
    venv = norm.venv
    n_envs = venv.num_envs
    ext = model.policy.mlp_extractor
    nb = ext.n_brain
    n_smell = getattr(ext.core, "n_smell", 0)
    rng = np.random.default_rng(seed)
    mem = hasattr(model.policy, "lstm_actor")      # recurrent fly brain (memory)
    n_neur = ext.core.n
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
        # (the statistics that go with the parent's BEST brain, which is the one copied: the latest ones had drifted)
        _vn = "best_vecnormalize.pkl" if os.path.exists(os.path.join(parent_dir, "best_vecnormalize.pkl")) else "vecnormalize.pkl"
        with open(os.path.join(parent_dir, _vn), "rb") as f:
            parent = pickle.load(f)
        # eyes + body sense: the parent's statistics (the parent may have had privileged inputs after
        # them; a grown brain has new smell inputs after them, measured in round 1 like the critic's)
        kb = nb - n_smell
        assert len(parent.obs_rms.mean) >= kb, f"parent brain has {len(parent.obs_rms.mean)} inputs, this one {kb}+{n_smell}"
        norm.obs_rms.mean[:kb], norm.obs_rms.var[:kb] = parent.obs_rms.mean[:kb], parent.obs_rms.var[:kb]
        norm.obs_rms.mean[kb:], norm.obs_rms.var[kb:] = 0.0, 1.0
        norm.obs_rms.count = parent.obs_rms.count
        norm.ret_rms = parent.ret_rms
        if remeasure is not None and len(remeasure):
            # inputs whose meaning changed from the parent (e.g. a camera instead of the wrist eye): their
            # statistics from a short teacher-driven warm-up, before any example is stored
            obs = venv.reset()
            raw = []
            for _ in range(max(1, 4000 // n_envs)):
                t_act = np.array(venv.env_method("teacher_action"), dtype=np.float32)
                obs, _, _, _ = venv.step(np.clip(t_act + rng.normal(0, teacher_noise, t_act.shape), -1, 1))
                raw.append(obs[:, remeasure].copy())
            raw = np.concatenate(raw)
            norm.obs_rms.mean[remeasure], norm.obs_rms.var[remeasure] = raw.mean(0), raw.var(0) + 1e-6
            log(f"[imitate] statistics of {len(remeasure)} changed inputs measured on {len(raw):,} teacher steps")
    norm.training = False                          # frozen while imitating (the student's inputs stay fixed)
    eval_norm.obs_rms = norm.obs_rms

    opt = torch.optim.Adam(_actor_params(model.policy), lr=lr)
    X = np.zeros((0, BLOCK, nb) if mem else (0, nb), np.float16)
    Y = np.zeros((0, BLOCK, 5) if mem else (0, 5), np.float32)     # teacher action (4) + gripper weight
    S = np.zeros((0, BLOCK), bool)                 # memory: episode starts (neuron state reset there)
    H = np.zeros((0, BLOCK // H_EVERY, n_neur), np.float16)   # memory: the student's neuron state, every H_EVERY steps
    data_f = os.path.join(out, "imitate_data.npz")
    if state["round"] > 0 and os.path.exists(data_f):      # the data collected so far (DAgger aggregates it)
        with np.load(data_f) as d:
            X, Y = d["X"], d["Y"]
            if Y.shape[-1] == 4:                   # (data from before the gripper weight)
                Y = np.concatenate([Y, np.ones(Y.shape[:-1] + (1,), np.float32)], -1)
            if mem:
                S, H = d["S"], d["H"]
        log(f"[imitate] data set restored: {len(X) * (BLOCK if mem else 1):,} examples")

    def evaluate():
        env = eval_norm.venv
        rets, lens, succ = [], [], []
        obs = env.reset()
        cur_r, cur_l = 0.0, 0
        h = torch.zeros(1, n_neur)
        while len(rets) < eval_episodes:
            o = torch.as_tensor(eval_norm.normalize_obs(obs)[:, :nb], dtype=torch.float32)
            with torch.no_grad():
                if mem:
                    a, h = _actor_step(model.policy, o, h)
                    a = a.clamp(-1, 1).numpy()
                else:
                    a = _actor_mean(model.policy, o).clamp(-1, 1).numpy()
            obs, r, done, info = env.step(a)
            if mem and done[0]:
                h = torch.zeros(1, n_neur)
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
        xs, ys, ss, ep_r, ep_s = [], [], [], [], []
        cur = np.zeros(n_envs)
        h = torch.zeros(n_envs, n_neur)            # memory: the student's neuron state follows its own perception
        start = np.ones(n_envs, bool)
        use_teacher = rng.random(n_envs) < beta
        hs = []
        priv_raw = []
        for j in range(n_steps // n_envs):
            if mem and j % H_EVERY == 0:           # state entering step j (already reset if an episode starts)
                hs.append(h.numpy().astype(np.float16))
            teacher = np.array(venv.env_method("teacher_action"), dtype=np.float32)
            nobs = norm.normalize_obs(obs)
            if k == 0:
                priv_raw.append(obs[:, nb - n_smell:].copy())
            with torch.no_grad():
                if mem:
                    student, h = _actor_step(model.policy, torch.as_tensor(nobs[:, :nb], dtype=torch.float32), h)
                    student = student.clamp(-1, 1).numpy()
                else:
                    student = _actor_mean(model.policy, torch.as_tensor(nobs[:, :nb], dtype=torch.float32)).clamp(-1, 1).numpy()
            if segment > 0:                        # driver decided per stretch (and at every episode start)
                new = start | (j % segment == 0)
                use_teacher = np.where(new, rng.random(n_envs) < beta, use_teacher)
            else:
                use_teacher = rng.random(n_envs) < beta
            # a little noise on whoever drives, so the data covers slightly-off situations too
            drive = np.where(use_teacher[:, None], teacher, student)
            act = np.clip(drive + rng.normal(0, teacher_noise, drive.shape), -1, 1)
            xs.append(nobs[:, :nb].astype(np.float16))
            grip = np.array(venv.get_attr("_grip_ctrl"), dtype=np.float32)
            decide = (teacher[:, 3] < 0) & (grip >= 40)        # the teacher closes a hand that is still open
            ys.append(np.c_[teacher, np.where(decide, grip_w, 1.0).astype(np.float32)])
            ss.append(start.copy())
            obs, r, done, info = venv.step(act)
            start = done.astype(bool)
            if mem and done.any():
                h[torch.as_tensor(done.astype(bool))] = 0.0
            cur += r
            for i in np.where(done)[0]:
                ep_r.append(cur[i]); ep_s.append(float(info[i].get("is_success", False))); cur[i] = 0
        if k == 0:                                 # critic inputs (and new smell inputs): statistics measured now
            pr = np.concatenate(priv_raw)
            norm.obs_rms.mean[nb - n_smell:], norm.obs_rms.var[nb - n_smell:] = pr.mean(0), pr.var(0) + 1e-6
        state["steps"] += n_steps
        if mem:                                    # per arm: consecutive steps, cut into stretches of BLOCK steps
            xa, ya, sa = np.stack(xs, 1), np.stack(ys, 1), np.stack(ss, 1)     # (arm, time, ...)
            nblk = xa.shape[1] // BLOCK
            cut = lambda a: a[:, :nblk * BLOCK].reshape(a.shape[0] * nblk, BLOCK, *a.shape[2:])  # noqa: E731
            xs, ys, S = [cut(xa)], [cut(ya)], np.concatenate([S, cut(sa)])
            ha = np.stack(hs, 1)[:, :nblk * (BLOCK // H_EVERY)]
            H = np.concatenate([H, ha.reshape(ha.shape[0] * nblk, BLOCK // H_EVERY, n_neur)])
        X = np.concatenate([X] + xs)
        Y = np.concatenate([Y] + ys)
        lim = cap // BLOCK if mem else cap
        if len(X) > lim:                           # keep a random subset of the old data + all new data
            new = len(xs[0]) if mem else n_steps
            keep_old = rng.choice(len(X) - new, lim - new, replace=False)
            idx = np.concatenate([np.sort(keep_old), np.arange(len(X) - new, len(X))])
            X, Y = X[idx], Y[idx]
            if mem:
                S, H = S[idx], H[idx]
        n_ex = len(X) * (BLOCK if mem else 1)
        t_collect = time.time() - t0
        # ---- 2. train the fly brain on the teacher's moves --------------------------------------
        log(f"[imitate] round {k + 1}/{iters}: collected in {t_collect:.0f}s; learning from {n_ex:,} examples ...")
        model.policy.train()
        # round 0 learns the teacher thoroughly; later rounds a fixed, sufficient amount
        per_up = N_CHUNKS * CHUNK if mem else batch
        n_up = int(first_epochs * n_ex / per_up) if k == 0 else min(max_updates, int(epochs * n_ex / per_up))
        n_up = max(1, n_up)
        losses = []
        for u in range(n_up):
            if mem:
                loss = _sequence_loss(model.policy, X, Y, S, H, rng)
            else:
                b = rng.integers(0, len(X), batch)
                pred = _actor_mean(model.policy, torch.as_tensor(X[b], dtype=torch.float32))
                loss = _weighted_err(pred, torch.as_tensor(Y[b]))
            if not torch.isfinite(loss):           # never let one bad batch poison the brain
                losses.append(float("nan"))
                continue
            opt.zero_grad()
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in _actor_params(model.policy) if p.grad is not None):
                continue
            torch.nn.utils.clip_grad_norm_(_actor_params(model.policy), 1.0)
            opt.step()
            losses.append(loss.item())
            if (u + 1) % max(1, n_up // 4) == 0 and u + 1 < n_up:
                log(f"[imitate] round {k + 1}/{iters}: learning {100 * (u + 1) // n_up}% "
                    f"(imitation loss {np.nanmean(losses[-50:]):.4f})")
        model.policy.eval()
        t_train = time.time() - t0 - t_collect
        # ---- 3. test the student alone ----------------------------------------------------------
        log(f"[imitate] round {k + 1}/{iters}: testing the student alone on {eval_episodes} episodes ...")
        rets, lens, succ = evaluate()
        hist["timesteps"] = np.append(hist["timesteps"], state["steps"])
        for key, v in (("results", rets), ("ep_lengths", lens), ("successes", succ)):
            hist[key] = np.concatenate([hist[key], v[None]])
        np.savez(ev_file, **hist)
        score = 1000.0 * float(succ.mean()) + float(rets.mean())   # most solved first; return breaks ties
        if score > state["best"]:
            state["best"] = score
            state["best_round"] = k + 1
            model.save(os.path.join(out, "best_model.zip"))
            norm.save(os.path.join(out, "best_vecnormalize.pkl"))    # the inputs' scaling at this moment
            torch.save(model.policy.state_dict(), os.path.join(out, "imitate_best_policy.pt"))
        log(f"[progress] {state['steps']:,} steps | training episodes: return {np.mean(ep_r) if ep_r else 0:.2f} | "
            f"solved {100 * np.mean(ep_s) if ep_s else 0:.0f}%")
        log(f"Eval num_timesteps={state['steps']}, episode_reward={rets.mean():.2f} +/- {rets.std():.2f}")
        log(f"Success rate: {100 * succ.mean():.2f}%")
        log(f"[imitate] round {k + 1}/{iters}: teacher drives {beta:.0%} | data {n_ex:,} | imitation loss "
            f"{np.nanmean(losses[-50:]):.4f} | student alone: return {rets.mean():.1f}, solved {succ.mean():.0%} | "
            f"{t_collect:.0f}s collect, {t_train:.0f}s learn")
        # ---- 4. checkpoint ----------------------------------------------------------------------
        state["round"] = k + 1
        torch.save(model.policy.state_dict(), os.path.join(out, "imitate_policy.pt"))
        with open(data_f + ".tmp", "wb") as f:                # the aggregated data, so a restart keeps it
            np.savez(f, X=X, Y=Y, **({"S": S, "H": H} if mem else {}))
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
    log(f"[imitate] done: {state['steps']:,} steps, best student score {state['best']:.1f} (solved x 1000 + return); "
        f"handing over to PPO with action std {final_std}")
    return model


def _sequence_loss(policy, X, Y, S, H, rng):
    """Memory: imitation error on N_CHUNKS pieces of CHUNK consecutive steps. Each piece starts from the
    neuron state saved during collection and first runs BURN warm-up steps without learning: the saved
    state comes from an older brain, the warm-up brings it up to date (R2D2's stored state + burn-in)."""
    blk = rng.integers(0, len(X), N_CHUNKS)
    k = rng.integers(0, (BLOCK - BURN - CHUNK) // H_EVERY + 1, N_CHUNKS)    # start at a saved state
    t0 = k * H_EVERY
    rows = t0[:, None] + np.arange(BURN + CHUNK)[None]
    x = torch.as_tensor(X[blk[:, None], rows], dtype=torch.float32)      # (pieces, time, inputs)
    y = torch.as_tensor(Y[blk[:, None], rows[:, BURN:]])
    keep = torch.as_tensor(1.0 - S[blk[:, None], rows], dtype=torch.float32)   # 0 where an episode starts
    h = torch.as_tensor(H[blk, k], dtype=torch.float32)
    with torch.no_grad():
        for t in range(BURN):
            _, h = _actor_step(policy, x[:, t], h * keep[:, t:t + 1])
    preds = []
    for t in range(BURN, BURN + CHUNK):
        a, h = _actor_step(policy, x[:, t], h * keep[:, t:t + 1])
        preds.append(a)
    return _weighted_err(torch.stack(preds, 1), y)


def _weighted_err(pred, y):
    """Mean squared imitation error; the gripper's error weighted by y[..., 4] (see run_dagger's grip_w)."""
    err = (pred - y[..., :4]) ** 2
    err = torch.cat([err[..., :3], err[..., 3:] * y[..., 4:5]], -1)
    return err.mean()
