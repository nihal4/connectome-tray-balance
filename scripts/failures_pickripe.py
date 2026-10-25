"""Why does a panda_pickripe brain fail? Usage: python scripts/failures_pickripe.py RUN [N]   (BRAIN=last: model.zip)

On N new layouts (the same as precision_pickripe's clean test): failures by cause, and every tomato's FIRST bruise
by what did it -- which tomato (green / turning / ripe), its role (the one being picked, held in the hand, or a
neighbour), the moment (approaching it, closing on it, pulling it off, carrying it, coming back), and which part of
the robot -- or a hard landing in the basket."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, '.')
import torch  # noqa: E402

torch.set_num_threads(2)
import mujoco  # noqa: E402

import ctb.envs.panda as P  # noqa: E402
from ctb.envs import build_env  # noqa: E402
from scripts.evaluate import load, make_norm, run_env_kw  # noqa: E402

run = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
best = os.environ.get("BRAIN", "best") != "last"
model, vn = load(run, best=best), make_norm(run, {}, best=best)
env = build_env('panda_pickripe', 0, **run_env_kw(run))
STAGE = ("green", "turning", "ripe")
causes, bruises = Counter(), Counter()


def phase():
    k = env._tgt
    if getattr(env, "_t_home", False):
        return "coming back"
    if env.grasped_tom(k):
        return "pulling it off" if env.att[k] else "carrying it"
    if env.att[k]:
        return "closing on it" if env._grip_ctrl < 120 else "approaching"
    return "after the drop"


orig = env._damage_check


def damage_check():
    before = list(env._bruised)
    ph = phase()
    hit = orig()
    new = [k for k in range(env.n_tom) if env._bruised[k] and not before[k]]
    if new:
        m, d, f = env.model, env.data, np.zeros(6)
        seen = set()
        for i in range(d.ncon):
            c = d.contact[i]
            for k in new:
                g = env.tom_geoms[k]
                if g in (c.geom1, c.geom2) and k not in seen:
                    o = c.geom2 if c.geom1 == g else c.geom1
                    if m.geom_bodyid[o] in env.robot_bodies:
                        mujoco.mj_contactForce(m, d, i, f)
                        if abs(f[0]) > P.DAMAGE_FORCE:
                            role = "held" if env.grasped_tom(k) else ("being picked" if k == env._tgt else "neighbour")
                            bruises[(STAGE[env.ripe[k]], role, ph, m.body(m.geom_bodyid[o]).name)] += 1
                            seen.add(k)
        for k in new:
            if k not in seen:
                bruises[("hard landing in the basket",)] += 1
    return hit


env._damage_check = damage_check
solved = 0
for i in range(N):
    obs, _ = env.reset(seed=140_000 + i)
    model.reset_memory()
    for t in range(env.max_steps):
        a, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
        obs, r, te, tr, info = env.step(a[0])
        if info['is_success'] or te or tr:
            break
    if info['is_success']:
        solved += 1
        continue
    if info.get('sim_unstable'):
        why = 'a tomato flung / dropped far'
    elif info['unripe_picked']:
        why = 'picked an unripe one'
    elif info['bruised']:
        why = 'bruised a tomato'
    elif info['ripe_picked'] > info['ripe_in_basket']:
        why = 'ripe one picked but not in the basket'
    elif info['ripe_in_basket'] == 0:
        why = 'never picked the first ripe one'
    else:
        why = 'never picked the second ripe one'
    causes[why] += 1
print(f"brain: {'best_model.zip' if best else 'model.zip'} | {N} layouts | solved {solved / N:.0%}")
print("failures by cause:")
for k, v in causes.most_common():
    print(f"   {v:3d}  {k}")
print("first bruises (tomato, its role, the moment, robot part):")
for k, v in bruises.most_common(15):
    print(f"   {v:3d}  {' | '.join(k)}")
