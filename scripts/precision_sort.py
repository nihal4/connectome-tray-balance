"""Colour sorting check (panda_sort): where does each colour end up?

Usage: python scripts/precision_sort.py RUN OUT.json [N]
N new random layouts (evaluation conditions). Per episode: block colour, solved, which bin the
block ended in (left / centre / right / none), whether it was grasped and released, and the
block-to-bin-centre distance when placed. Prints a colour x bin table (the confusion matrix).
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, '.')
import torch  # noqa: E402

torch.set_num_threads(1)
from ctb.envs import build_env  # noqa: E402
from ctb.envs.panda import BIN_IN, REST_Z, SORT_COLOURS  # noqa: E402
from scripts.evaluate import load, make_norm, run_env_kw  # noqa: E402

run, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
model = load(run, best=True)
vn = make_norm(run, {}, best=True)
env = build_env('panda_sort', 0, **run_env_kw(run))
BINS = ['left', 'centre', 'right']


def which_bin():
    b = env.block_pos()
    for k, mc in enumerate(env.bin_mocaps):
        if np.all(np.abs(b[:2] - env.data.mocap_pos[mc][:2]) < BIN_IN) and b[2] < REST_Z + 0.012:
            return k
    return -1


res = []
t0 = time.time()
for i in range(N):
    obs, _ = env.reset(seed=60_000 + i)
    colour = env._colour
    t_succ, grasped = None, False
    for t in range(env.max_steps):
        a, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
        obs, r, te, tr, info = env.step(a[0])
        grasped |= env.is_grasped()
        if info['is_success'] and t_succ is None:
            t_succ = t + 1
    k = which_bin()
    c = env.data.mocap_pos[env.bin_mocaps[k]] if k >= 0 else None
    res.append(dict(seed=60_000 + i, colour=SORT_COLOURS[colour][0], success=bool(info['is_success']),
                    t_success=t_succ, grasped=bool(grasped), end_bin=BINS[k] if k >= 0 else 'none',
                    released=not env.block_touched(),
                    off_cm=float(100 * np.linalg.norm(env.block_pos()[:2] - c[:2])) if c is not None else None))
    if (i + 1) % 50 == 0:
        print(i + 1, 'episodes %.0fs' % (time.time() - t0), flush=True)
json.dump(res, open(out, 'w'))
print(f'{len(res)} episodes, solved {np.mean([r["success"] for r in res]):.0%}')
print('colour   n   solved | ended in:  left centre right none')
for name, _, _ in SORT_COLOURS:
    rs = [r for r in res if r['colour'] == name]
    cnt = [sum(r['end_bin'] == b for r in rs) for b in BINS + ['none']]
    print(f'{name:7s} {len(rs):3d}  {np.mean([r["success"] for r in rs]):5.0%}  |            ' + '  '.join(f'{c:4d}' for c in cnt))
