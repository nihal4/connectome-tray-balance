"""Precision of a lift brain (used for results/v10_panda_lift_imitate): same rules for every brain (goal rule: block within 2.5 cm of the
point 8 cm above its start, held 5 steps). Usage: python scripts/precision_lift.py RUN OUT.json [random|map] [N]"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")      # off-screen rendering (no display needed)
import sys, json, time
import numpy as np
import mujoco
sys.path.insert(0, '.')
torch_threads = 1
import torch; torch.set_num_threads(1)
from ctb.envs import build_env
from ctb.envs.panda import BLOCK_LO, BLOCK_HI, LIFT_GOAL, GOAL_DIST, LIFT_HOLD
from scripts.evaluate import load, make_norm, run_env_kw

run, out, mode = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 100
model = load(run, best=True)
vn = make_norm(run, {}, best=True)
env = build_env('panda_lift', 0, **run_env_kw(run))


def place(xy):
    d = env.data
    for qa in env.decoy_q:
        d.qpos[qa:qa + 3] = [0, 0, -5]
    d.qpos[env.block_q:env.block_q + 2] = xy
    d.qvel[:] = 0
    mujoco.mj_forward(env.model, d)
    for _ in range(50):
        mujoco.mj_step(env.model, d)
    for qa in env.decoy_q:
        d.qpos[qa:qa + 3] = [0, 0, -5]
    d.qvel[:] = 0
    mujoco.mj_forward(env.model, d)
    env._block_z0 = float(env.block_pos()[2])
    env._goal = env.block_pos() + np.array([0, 0, LIFT_GOAL])
    env._prev_img = None
    return env._get_obs()


def episode(seed, xy=None):
    obs, _ = env.reset(seed=seed)
    model.reset_memory()
    if xy is not None:
        obs = place(xy)
    start = env.block_pos()
    goal = env._goal.copy()
    hold = 0
    t_succ = None
    grasped = False
    path = []
    for t in range(env.max_steps):
        a, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
        obs, r, te, tr, info = env.step(a[0])
        b = env.block_pos()
        g = float(np.linalg.norm(b - goal))
        held = env.is_grasped()
        grasped |= held
        hold = hold + 1 if (held and g < GOAL_DIST) else 0
        if hold >= LIFT_HOLD and t_succ is None:
            t_succ = t + 1
        path.append(b)
    path = np.array(path)
    b = path[-1]
    lifted = path[:, 2] - start[2]
    return dict(seed=int(seed), x=float(start[0]), y=float(start[1]),
                success=t_succ is not None, t_success=t_succ, grasped=bool(grasped),
                max_lift_cm=float(100 * lifted.max()),
                end_goal_err_cm=float(100 * np.linalg.norm(b - goal)),
                end_lift_cm=float(100 * (b[2] - start[2])),
                end_side_cm=float(100 * np.linalg.norm(b[:2] - start[:2])),
                # largest sideways offset while the block is in the air (> 2 cm up)
                air_side_cm=float(100 * max([np.linalg.norm(p[:2] - start[:2]) for p in path if p[2] - start[2] > 0.02] or [0])),
                held_at_end=bool(env.is_grasped() and np.linalg.norm(b - goal) < GOAL_DIST))


res = []
t0 = time.time()
if mode == 'random':
    for i in range(N):
        res.append(episode(20_000 + i))
        if (i + 1) % 10 == 0:
            print(i + 1, 'episodes %.0fs' % (time.time() - t0), flush=True)
            json.dump(res, open(out + '.part', 'w'))
else:                                     # 5 x 5 grid, N starts each, decoys removed
    xs = np.linspace(BLOCK_LO[0], BLOCK_HI[0], 5)
    ys = np.linspace(BLOCK_LO[1], BLOCK_HI[1], 5)
    for x in xs:
        for y in ys:
            for i in range(N):
                res.append(episode(30_000 + 97 * i + int(1000 * x) + int(100 * (y + 1)), (x, y)))
        print('row x=%.2f done %.0fs' % (x, time.time() - t0), flush=True)
json.dump(res, open(out, 'w'))
s = np.mean([r['success'] for r in res])
print(run, mode, len(res), 'episodes, success %.0f%%' % (100 * s), '%.0fs' % (time.time() - t0))
