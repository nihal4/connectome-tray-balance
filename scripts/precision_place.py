"""Placement precision of a pick-and-place brain (panda_place).

Usage: python scripts/precision_place.py RUN OUT.json [random|map] [N]
  random: N new random layouts (blocks, bins and decoys as in evaluation)
  map:    the BIN on a 5 x 5 grid across its area, N random block / start layouts per cell
Per episode: solved (block resting in the bin, released, 0.2 s), when, whether it was grasped
and carried, and where the block ended relative to the bin centre.
"""
import json
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, '.')
import torch  # noqa: E402

torch.set_num_threads(1)
from ctb.envs import build_env  # noqa: E402
from ctb.envs.panda import BIN_LO, BIN_HI, BIN_MIN_GAP, REST_Z, BLOCK_LO, BLOCK_HI  # noqa: E402
from scripts.evaluate import load, make_norm, run_env_kw  # noqa: E402

run, out, mode = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 100
model = load(run, best=True)
vn = make_norm(run, {})
env = build_env('panda_place', 0, **run_env_kw(run))


def move_bin(xy, rng):
    """Put the bin at xy; move the block (and decoys) away from it if needed."""
    d = env.data
    d.mocap_pos[env.bin_mocap] = [xy[0], xy[1], 0.0]
    blk = env.block_pos()[:2]
    if np.linalg.norm(blk - xy) <= BIN_MIN_GAP:
        for _ in range(200):
            blk = rng.uniform(BLOCK_LO, BLOCK_HI)
            if np.linalg.norm(blk - xy) > BIN_MIN_GAP:
                break
        d.qpos[env.block_q:env.block_q + 2] = blk
    for qa in env.decoy_q:                         # keep decoys out of the bin
        if d.qpos[qa + 2] > -1 and np.linalg.norm(d.qpos[qa:qa + 2] - xy) < 0.13:
            d.qpos[qa:qa + 3] = [0, 0, -5]
    d.qvel[:] = 0
    mujoco.mj_forward(env.model, d)
    for _ in range(50):
        mujoco.mj_step(env.model, d)
    env._block_z0 = float(env.block_pos()[2])
    env._goal = env.bin_pos() + np.array([0, 0, REST_Z])
    env._prev_img = None
    return env._get_obs()


def episode(seed, bin_xy=None):
    obs, _ = env.reset(seed=seed)
    if bin_xy is not None:
        obs = move_bin(bin_xy, np.random.default_rng(seed))
    c = env.bin_pos()
    start = env.block_pos()
    t_succ, grasped, carried, max_h = None, False, False, 0.0
    for t in range(env.max_steps):
        a, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
        obs, r, te, tr, info = env.step(a[0])
        b = env.block_pos()
        g = env.is_grasped()
        grasped |= g
        carried |= g and np.linalg.norm(b[:2] - c[:2]) < 0.05
        max_h = max(max_h, b[2] - start[2])
        if info['is_success'] and t_succ is None:
            t_succ = t + 1
    b = env.block_pos()
    return dict(seed=int(seed), bin_x=float(c[0]), bin_y=float(c[1]), block_x=float(start[0]), block_y=float(start[1]),
                success=bool(info['is_success']), t_success=t_succ, grasped=bool(grasped), carried=bool(carried),
                in_bin=bool(env.in_bin()), released=not env.block_touched(),
                end_off_cm=float(100 * np.linalg.norm(b[:2] - c[:2])), end_z_cm=float(100 * b[2]),
                max_lift_cm=float(100 * max_h))


res = []
t0 = time.time()
if mode == 'random':
    for i in range(N):
        res.append(episode(40_000 + i))
        if (i + 1) % 20 == 0:
            print(i + 1, 'episodes %.0fs' % (time.time() - t0), flush=True)
else:
    xs = np.linspace(BIN_LO[0], BIN_HI[0], 5)
    ys = np.linspace(BIN_LO[1], BIN_HI[1], 5)
    for x in xs:
        for y in ys:
            for i in range(N):
                res.append(episode(50_000 + 97 * i + int(1000 * x) + int(100 * (y + 1)), np.array([x, y])))
        print('row x=%.2f done %.0fs' % (x, time.time() - t0), flush=True)
json.dump(res, open(out, 'w'))
print(run, mode, len(res), 'episodes, solved %.0f%%' % (100 * np.mean([r['success'] for r in res])),
      '%.0fs' % (time.time() - t0))
