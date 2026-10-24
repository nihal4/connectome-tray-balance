"""Greenhouse phase 1 (panda_harvest): how well, how fast and how gently does the brain pick?

Usage: python scripts/precision_harvest.py RUN OUT.json [N] [N_SMELL]
N new layouts (best brain): solved, picked (stalk snapped), dropped, steps to pick / to basket, steps
touching the plant. Then N_SMELL layouts for each smell probe, to see what the brain uses smell for:
  no_smell        antennae silenced
  smell_shift_6cm the smell comes from 6 cm beside the tomato (left or right, half each)
  smell_constant  full-strength smell everywhere (no rise as the hand gets close)
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")      # off-screen rendering (no display needed)
import json
import sys
import time
import types

import numpy as np

sys.path.insert(0, '.')
import torch  # noqa: E402

torch.set_num_threads(1)
import ctb.envs.panda as P  # noqa: E402
from ctb.envs import build_env  # noqa: E402
from scripts.evaluate import load, make_norm, run_env_kw  # noqa: E402

run, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 200
NS = int(sys.argv[4]) if len(sys.argv) > 4 else 50
model = load(run, best=True)
vn = make_norm(run, {}, best=True)


def probe_smell(shift=None, const=False):
    """Replacement smell_obs: the smell source moved by shift (per episode), or constant strength."""
    def smell_obs(self):
        tcp, src = self.tcp_pos(), self.block_pos() + (self._shift if shift is not None else 0)
        res = []
        for a in P.SMELL_ANTENNAE:
            d2 = float(np.sum((tcp + a - src) ** 2))
            conc = (1.0 if const else np.exp(-d2 / (2 * P.SMELL_SIGMA ** 2))) * np.exp(self.rng.normal(0, P.SMELL_FLICKER)) + 0.02
            drive = conc * self.smell_drive[self._fruit]
            res.append(np.tanh(2.0 * drive) + self.rng.normal(0, P.SMELL_NOISE, drive.shape))
        return np.concatenate(res)
    return smell_obs


def episodes(n, seed0, smell=None, shift=None, **kw):
    env = build_env('panda_harvest', 0, **run_env_kw(run), **kw)
    if smell is not None:
        env.smell_obs = types.MethodType(smell, env)
    res = []
    for i in range(n):
        env._shift = np.array([0, (1 if i % 2 else -1) * shift, 0]) if shift else np.zeros(3)
        obs, _ = env.reset(seed=seed0 + i)
        model.reset_memory()
        t_pick = t_done = None
        closest, cxyz = 9.0, None
        for t in range(env.max_steps):
            a, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
            obs, r, te, tr, info = env.step(a[0])
            if env.attached and info['dist'] < closest:
                closest, cxyz = info['dist'], (env.tcp_pos() - env.block_pos())
            if info['picked'] and t_pick is None:
                t_pick = t + 1
            if info['is_success'] and t_done is None:
                t_done = t + 1
                break
        dropped = bool(info['picked'] and not info['is_success'] and not env.is_grasped())
        res.append(dict(seed=seed0 + i, success=bool(info['is_success']), picked=bool(info['picked']),
                        dropped=dropped, steps_to_pick=t_pick, steps_to_basket=t_done,
                        plant_touch_steps=int(info['plant_touch_steps']), closest_on_vine=round(closest, 4),
                        offset_y_at_closest=None if cxyz is None else round(float(cxyz[1]), 4)))
    return res


def summary(res, title):
    s = lambda k: [r[k] for r in res if r[k] is not None]   # noqa: E731
    print(f"{title}: {len(res)} layouts | solved {np.mean(s('success')):.0%} | picked {np.mean(s('picked')):.0%} | "
          f"dropped {np.mean(s('dropped')):.0%} | steps to pick median {np.median(s('steps_to_pick')) if s('steps_to_pick') else '-'} | "
          f"to basket median {np.median(s('steps_to_basket')) if s('steps_to_basket') else '-'} | plant touched: "
          f"{np.mean([r['plant_touch_steps'] > 0 for r in res]):.0%} of episodes, mean {np.mean(s('plant_touch_steps')):.1f} steps | "
          f"closest on the vine median {np.median(s('closest_on_vine')) * 100:.1f} cm", flush=True)


t0 = time.time()
out_d = {'clean': episodes(N, 120_000)}
summary(out_d['clean'], 'clean')
for name, kw in [('no_smell', dict(smell_off=True)), ('smell_shift_6cm', dict(smell=probe_smell(shift=0.06), shift=0.06)),
                 ('smell_constant', dict(smell=probe_smell(const=True)))]:
    out_d[name] = episodes(NS, 125_000, **kw)
    summary(out_d[name], name)
json.dump(out_d, open(out, 'w'))
print(f'{time.time() - t0:.0f}s')
