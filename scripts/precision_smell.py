"""Ripe or rotten (panda_smell): where does each fruit end up, with and without smell?

Usage: python scripts/precision_smell.py RUN OUT.json [N] [N_NO_SMELL]
N new layouts with smell, N_NO_SMELL with the antennae silenced. Per episode: fruit, solved, which
bin it ended in (ripe = left, rotten = right, none), grasped. Prints fruit x bin tables.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, '.')
import torch  # noqa: E402

torch.set_num_threads(1)
from ctb.envs import build_env  # noqa: E402
from ctb.envs.panda import BIN_IN, REST_Z  # noqa: E402
from scripts.evaluate import load, make_norm, run_env_kw  # noqa: E402

run, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
N0 = int(sys.argv[4]) if len(sys.argv) > 4 else 100
model = load(run, best=True)
vn = make_norm(run, {}, best=True)
BINS = ['ripe bin', 'rotten bin']


def episodes(n, seed0, **kw):
    env = build_env('panda_smell', 0, **run_env_kw(run), **kw)
    res = []
    for i in range(n):
        obs, _ = env.reset(seed=seed0 + i)
        fruit = env._fruit
        grasped, t_succ = False, None
        for t in range(env.max_steps):
            a, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
            obs, r, te, tr, info = env.step(a[0])
            grasped |= env.is_grasped()
            if info['is_success'] and t_succ is None:
                t_succ = t + 1
        b = env.block_pos()
        end = 'none'
        for k, mc in enumerate(env.bin_mocaps):
            if np.all(np.abs(b[:2] - env.data.mocap_pos[mc][:2]) < BIN_IN) and b[2] < REST_Z + 0.012:
                end = BINS[k]
        res.append(dict(seed=seed0 + i, fruit=('ripe', 'rotten')[fruit], success=bool(info['is_success']),
                        end_bin=end, grasped=bool(grasped), t_success=t_succ))
    return res


def table(res, title):
    print(f'{title}: {len(res)} episodes, solved {np.mean([r["success"] for r in res]):.0%}')
    print('fruit    n   solved | ended in: ripe bin  rotten bin  none')
    for f in ('ripe', 'rotten'):
        rs = [r for r in res if r['fruit'] == f]
        cnt = [sum(r['end_bin'] == b for r in rs) for b in BINS + ['none']]
        print(f'{f:7s} {len(rs):3d}  {np.mean([r["success"] for r in rs]):5.0%}  |          ' + '  '.join(f'{c:8d}' for c in cnt))


t0 = time.time()
with_smell = episodes(N, 70_000)
table(with_smell, 'with smell')
no_smell = episodes(N0, 75_000, smell_off=True)
table(no_smell, 'antennae silenced')
json.dump({'with_smell': with_smell, 'no_smell': no_smell}, open(out, 'w'))
print(f'{time.time() - t0:.0f}s')
