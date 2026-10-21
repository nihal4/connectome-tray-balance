"""Success at fixed brightness levels (Panda tasks): where does a brain hold up, where does it break?

Usage: python scripts/light_sweep.py RUN OUT.json [N] [factors...]
Each factor (default 0.4 0.5 0.7 1 1.4 2 2.5) is used for every one of N new layouts (the same
layouts for every factor), best brain, evaluation conditions otherwise.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, '.')
import torch  # noqa: E402

torch.set_num_threads(1)
from ctb.envs import build_env  # noqa: E402
from scripts.evaluate import load, make_norm, run_config, run_env_kw  # noqa: E402

run, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 40
factors = [float(f) for f in sys.argv[4:]] or [0.4, 0.5, 0.7, 1.0, 1.4, 2.0, 2.5]
model = load(run, best=True)
vn = make_norm(run, {}, best=True)
env_name = run_config(run)["env"]
res = {}
t0 = time.time()
for f in factors:
    env = build_env(env_name, 0, **run_env_kw(run), light_choices=(f,))
    ok = 0
    for i in range(N):
        obs, _ = env.reset(seed=80_000 + i)
        model.reset_memory()
        for t in range(env.max_steps):
            a, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
            obs, r, te, tr, info = env.step(a[0])
        ok += bool(info['is_success'])
    res[str(f)] = ok / N
    print(f'brightness x{f:<4}  solved {ok}/{N} ({100 * ok / N:.0f} %)   {time.time() - t0:.0f}s', flush=True)
json.dump({'run': run, 'episodes': N, 'solved': res}, open(out, 'w'), indent=1)
