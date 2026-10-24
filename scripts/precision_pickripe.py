"""Greenhouse phase 2 (panda_pickripe): does the brain pick only the ripe tomatoes, and how does it tell?

Usage: python scripts/precision_pickripe.py RUN OUT.json [N] [N_TEST]
N new layouts (best brain): per episode the ripeness of the three tomatoes, how many ripe ones ended in the
basket, unripe ones picked, steps, plant contact. Then N_TEST layouts for each probe:
  no_smell       antennae silenced (colour only)
  all_look_ripe  every tomato looks ripe to the eyes (smell only)
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")      # off-screen rendering (no display needed)
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, '.')
import torch  # noqa: E402

torch.set_num_threads(1)
from ctb.envs import build_env  # noqa: E402
from ctb.envs.panda import RIPENESS  # noqa: E402
from scripts.evaluate import load, make_norm, run_env_kw  # noqa: E402

run, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 200
NT = int(sys.argv[4]) if len(sys.argv) > 4 else 50
BEST = os.environ.get("BRAIN", "best") != "last"        # BRAIN=last: the latest brain (model.zip) instead of the best
model = load(run, best=BEST)
vn = make_norm(run, {}, best=BEST)
print("brain:", "best_model.zip" if BEST else "model.zip (latest)", flush=True)


def episodes(n, seed0, **kw):
    env = build_env('panda_pickripe', 0, **run_env_kw(run), **kw)
    res = []
    for i in range(n):
        obs, _ = env.reset(seed=seed0 + i)
        model.reset_memory()
        ripeness = [RIPENESS[r] for r in env.ripe]
        t_done, touch_steps = None, 0
        for t in range(env.max_steps):
            a, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
            obs, r, te, tr, info = env.step(a[0])
            touch_steps += env._touching_others()          # touching a tomato it is not picking (any force)
            if info['is_success']:
                t_done = t + 1
                break
        res.append(dict(seed=seed0 + i, ripeness=ripeness, n_ripe=info['n_ripe'], success=bool(info['is_success']),
                        ripe_in_basket=int(info['ripe_in_basket']), ripe_picked=int(info['ripe_picked']),
                        unripe_picked=int(info['unripe_picked']), steps=t_done,
                        plant_touch_steps=int(info['plant_touch_steps']),
                        bruised=int(info.get('bruised', 0)), damage_steps=int(info.get('damage_steps', 0)),
                        touch_steps=touch_steps))
    return res


def summary(res, title):
    ripe = sum(r['n_ripe'] for r in res)
    unripe = sum(3 - r['n_ripe'] for r in res)
    st = [r['steps'] for r in res if r['steps']]
    print(f"{title}: {len(res)} layouts | solved {np.mean([r['success'] for r in res]):.0%} | ripe tomatoes into the basket "
          f"{sum(r['ripe_in_basket'] for r in res)}/{ripe} | unripe picked {sum(r['unripe_picked'] for r in res)}/{unripe} "
          f"(in {np.mean([r['unripe_picked'] > 0 for r in res]):.0%} of layouts) | tomatoes bruised {sum(r.get('bruised', 0) for r in res)} "
          f"(in {np.mean([r.get('bruised', 0) > 0 for r in res]):.0%} of layouts) | steps median {np.median(st) if st else '-'} | touching other tomatoes: "
          f"{np.mean([r['touch_steps'] > 0 for r in res]):.0%} of layouts, mean {np.mean([r['touch_steps'] for r in res]):.1f} steps", flush=True)
    for k in (1, 2):
        sub = [r for r in res if r['n_ripe'] == k]
        if sub:
            print(f"   {k} ripe: {len(sub)} layouts, solved {np.mean([r['success'] for r in sub]):.0%}")


t0 = time.time()
out_d = {'clean': episodes(N, 140_000)}
summary(out_d['clean'], 'clean')
for name, kw in [('no_smell', dict(smell_off=True)), ('all_look_ripe', dict(tomato_look='all_ripe')), ('turning_smells_ripe', dict(smell_confuse=True))]:
    out_d[name] = episodes(NT, 145_000, **kw)
    summary(out_d[name], name)
json.dump(out_d, open(out, 'w'))
print(f'{time.time() - t0:.0f}s')
