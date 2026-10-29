"""Run a trained brain (phase 2c) and, at every step, ask the scripted teacher what it would do there and which
part of the teacher answers. Shows where a student stalls (e.g. V2: 41 % of steps the teacher said CLOSE and the
brain kept the hand open).

    python scripts/diag_student_vs_teacher.py RUN_DIR 40 out.pkl
"""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np
from multiprocessing import Pool

RUN = sys.argv[1]


def branch(env, P):
    """Which part of the teacher answers at this state (same tests, same order as the teacher)."""
    tcp = env.tcp_pos()
    if getattr(env, "_t_home", False):
        return "go home"
    if P._below_truss(env, tcp):
        return "below truss: back out"
    tom = env.stalk_point()
    if np.linalg.norm(tcp - tom) > P.LOOK_NEAR:
        return "look: go near"
    rel = tom - tcp
    ahead = float(rel @ P.APPROACH)
    perp = rel - ahead * P.APPROACH
    ey, eo = abs(perp[1]), float(np.hypot(perp[0], perp[2]))
    if env._grip_ctrl < 40:
        return "closing"
    if getattr(env, "_t_backoff", False) and ahead < P.PRE_GRASP - 0.01:
        return "after miss: back off"
    if ahead > P.PRE_GRASP + 0.02 or (ahead > 0.02 and np.linalg.norm(perp) > 0.03):
        return "to pre-grasp point"
    if ahead < -0.004 or (ahead < 0.015 and (ey > 3 * P.GRASP_Y_TOL or eo > 3 * P.GRASP_O_TOL)):
        return "too far in / off line: back out"
    if np.linalg.norm(rel) < P.GRASP_AT and np.linalg.norm(env.tcp_vel()) < 0.02:
        return "CLOSE"
    if np.linalg.norm(rel) < P.GRASP_AT:
        return "there, wait until still"
    return "servo in"


def one(seed):
    import evaluate as E
    from ctb.envs import panda as P
    model = E.load(RUN, best=True)
    vn = E.make_norm(RUN, {}, seed, best=True)
    env = vn.venv.envs[0].unwrapped
    obs = vn.reset()
    model.reset_memory()
    rows, info = [], {}
    for t in range(env.max_steps):
        a, _ = model.predict(obs, deterministic=True)
        a = np.clip(a[0], -1, 1)
        k = env._tgt
        on_vine = k is not None and k >= 0 and env.att[k] and not env.is_grasped()
        held = env.is_grasped()
        d = float(np.linalg.norm(env.stalk_point() - env.tcp_pos())) if on_vine else np.nan
        # the teacher's answer at this state (its internal timers advance as in DAgger labelling)
        br = branch(env, P) if on_vine else ("carry" if held else "other")
        ta = P.harvest_teacher(env)
        rows.append(dict(br=br, t=t, on_vine=on_vine, held=held, d=d, sa=a.tolist(), ta=ta.tolist(),
                         v=float(np.linalg.norm(env.tcp_vel())), w=float(env.gripper_width()),
                         grip=float(env._grip_ctrl), basket=int(info.get("ripe_in_basket", 0))))
        obs, r, done, infos = vn.step(a[None])
        info = infos[0]
        if done[0]:
            break
    return seed, bool(info.get("is_success")), info.get("n_ripe", 0), info.get("ripe_in_basket", 0), rows


def report(res):
    from collections import Counter, defaultdict
    print(f"solved {sum(r[1] for r in res)}/{len(res)}; ripe in basket {sum(r[3] for r in res)}/{sum(r[2] for r in res)}")
    allc, stuck_end = Counter(), Counter()
    st = defaultdict(list)
    for seed, ok, nr, nb, rows in res:
        for r in rows:
            allc[r["br"]] += 1
            sa, ta = np.array(r["sa"]), np.array(r["ta"])
            st[r["br"]].append((np.linalg.norm(sa[:3]), np.linalg.norm(ta[:3]),
                                float(sa[:3] @ ta[:3] / (np.linalg.norm(sa[:3]) * np.linalg.norm(ta[:3]) + 1e-9)), sa[3], ta[3], r["d"]))
        if not ok:
            tail = rows[-100:]                                   # what the teacher says over the last 4 s of a failure
            stuck_end.update(Counter(r["br"] for r in tail))
    n = sum(allc.values())
    print(f"\n{'teacher branch':34s} {'share of steps':>14s} {'|student xyz|':>13s} {'|teacher xyz|':>13s} {'same dir (cos)':>14s} {'grip s / t':>12s}")
    for b, c in allc.most_common():
        x = np.array(st[b])
        print(f"{b:34s} {c / n:13.1%} {x[:, 0].mean():13.2f} {x[:, 1].mean():13.2f} {x[:, 2].mean():14.2f} {x[:, 3].mean():6.2f}/{x[:, 4].mean():5.2f}")
    m = sum(stuck_end.values())
    print("\nfailed episodes, last 4 s: the teacher is saying")
    for b, c in stuck_end.most_common():
        print(f"   {c / m:6.1%}  {b}")


if __name__ == "__main__":
    n = int(sys.argv[2])
    with Pool(os.cpu_count()) as p:
        res = p.map(one, range(160_000, 160_000 + n))
    import pickle
    pickle.dump(res, open(sys.argv[3], "wb"))
    report(res)
