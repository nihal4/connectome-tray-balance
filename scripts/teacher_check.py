"""How good is the scripted harvest teacher? Runs it (true positions, no cameras) on N new phase-2c layouts in
parallel and reports solved %, by number of ripe tomatoes, and every failure by cause, with the seeds of the
failures (to replay one: --seeds S --video out.mp4).

    python scripts/teacher_check.py --n 1000                 # the full check
    python scripts/teacher_check.py --seeds 140007 --video /tmp/t.mp4
"""
import argparse
import os
import sys
from collections import Counter
from multiprocessing import Pool

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

KW = dict(task="pickripe", many=True, gentle=True, grip_mode="absolute", privileged=True, bruise_end=True)


def _env(kw):
    from ctb.envs import panda as P
    return P, P.FlyPandaEnv(seed=0, **{**KW, **kw})


def run_one(args):
    seed, kw, video = args
    P, env = _env(kw)
    env.reset(seed=seed)
    m = env.model
    frames, first_bruise = [], None
    info = {}
    for t in range(env.max_steps):
        before = list(env._bruised)
        _, _, te, tr, info = env.step(P.harvest_teacher(env))
        if video:
            frames.append(env.render())
        new = [k for k in range(env.n_tom) if env._bruised[k] and not before[k]]
        if new and first_bruise is None:
            k = new[0]
            d, f, part = env.data, np.zeros(6), "landing"
            for i in range(d.ncon):
                c = d.contact[i]
                if env.tom_geoms[k] in (c.geom1, c.geom2):
                    o = c.geom2 if c.geom1 == env.tom_geoms[k] else c.geom1
                    if m.geom_bodyid[o] in env.robot_bodies:
                        import mujoco
                        mujoco.mj_contactForce(m, d, i, f)
                        if abs(f[0]) > P.DAMAGE_FORCE:
                            part = m.body(m.geom_bodyid[o]).name
            role = "held" if env.grasped_tom(k) else ("target" if k == env._tgt else "neighbour")
            first_bruise = f"{P.RIPENESS[env.ripe[k]]} {role} by {part}"
        if info.get("is_success") or te or tr:
            break
    if info.get("is_success"):
        why = "solved"
    elif info.get("sim_unstable"):
        why = "tomato flung / sim unstable"
    elif info.get("unripe_picked"):
        why = "picked an unripe one"
    elif info.get("bruised"):
        why = "bruised: " + (first_bruise or "?")
    elif info["ripe_picked"] > info["ripe_in_basket"]:
        why = "ripe one picked but not in the basket"
    elif info["ripe_in_basket"] == 0:
        why = "never picked the first ripe one"
    else:
        why = "a later ripe one never picked"
    if video:
        import imageio
        imageio.mimsave(video, frames, fps=25)
    return seed, info.get("n_ripe", 0), why, t + 1, env.truss_draws, env.truss_rejects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--first_seed", type=int, default=140_000)
    ap.add_argument("--seeds", default=None, help="comma-separated seeds instead of --n")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--video", default=None, help="(one seed) save the episode as an mp4")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--no_filter", action="store_true", help="fully random trusses (no feasibility filter)")
    a = ap.parse_args()
    kw = {"max_steps": a.max_steps} if a.max_steps else {}
    if a.no_filter:
        kw["truss_filter"] = False
    if a.video:
        kw["render_mode"] = "rgb_array"
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(range(a.first_seed, a.first_seed + a.n))
    jobs = [(s, kw, a.video if a.video and len(seeds) == 1 else None) for s in seeds]
    if len(jobs) == 1:
        res = [run_one(jobs[0])]
    else:
        with Pool(a.workers) as p:
            res = p.map(run_one, jobs, chunksize=4)
    n = len(res)
    ok = [r for r in res if r[2] == "solved"]
    print(f"teacher: {len(ok)}/{n} solved = {len(ok) / n:.1%}   steps median {np.median([r[3] for r in ok]) if ok else 0:.0f}"
          f" (90th pct {np.percentile([r[3] for r in ok], 90) if ok else 0:.0f})")
    draws, rej = sum(r[4] for r in res), sum(r[5] for r in res)
    if draws:
        print(f"   truss filter: {rej} of {draws} random trusses rejected as unreachable ({rej / draws:.1%})")
    for k in (1, 2, 3):
        sub = [r for r in res if r[1] == k]
        if sub:
            print(f"   {k} ripe: {sum(r[2] == 'solved' for r in sub)}/{len(sub)} = {np.mean([r[2] == 'solved' for r in sub]):.0%}")
    fails = Counter(r[2] for r in res if r[2] != "solved")
    for why, c in fails.most_common():
        ex = [r[0] for r in res if r[2] == why][:5]
        print(f"   {c:4d}  {why}   e.g. seeds {ex}")


if __name__ == "__main__":
    main()
