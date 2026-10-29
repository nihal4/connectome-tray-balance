"""Is the teacher's 'close now' moment visible to the fly brain? (phase 2c)

Runs the harvest teacher on N new trusses with the brain's sensors (wrist + front depth cameras, smell, body),
keeps every step with the hand within 8 cm of the target tomato, and fits gradient-boosted trees (truss-grouped
5-fold CV) that predict, from the brain's own inputs only, (a) the teacher's close decision and (b) the hand-to-
tomato distance. Result (400 trusses): AUC 0.998, distance median error 0.3 mm -> the information is there.

    python scripts/diag_close_cue.py --n 400
"""
import argparse
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

KW = dict(task="pickripe", many=True, gentle=True, grip_mode="absolute", privileged=True, bruise_end=True,
          wrist_cam="rgbd", head_cam="rgbd", head_fovy=26, eye_adapt=True, smooth=True, reward_mode="events")


def run(seed):
    from ctb.envs import panda as P
    env = P.FlyPandaEnv(seed=0, **KW)
    obs, _ = env.reset(seed=seed)
    rows = []
    for _ in range(env.max_steps):
        a = P.harvest_teacher(env)
        k = env._tgt
        if k is not None and k >= 0 and env.att[k] and not env.is_grasped():
            d = float(np.linalg.norm(env.stalk_point() - env.tcp_pos()))
            if d < 0.08:
                rows.append((obs.astype(np.float32), float(a[3] < 0 and env._grip_ctrl >= 40), d, seed))
        obs, _, te, tr, _ = env.step(a)
        if te or tr:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--first_seed", type=int, default=150_000)
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier as C, HistGradientBoostingRegressor as R
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from ctb.envs.panda import n_priv
    with Pool(os.cpu_count()) as p:
        res = sum(p.map(run, range(a.first_seed, a.first_seed + a.n)), [])
    X = np.stack([r[0] for r in res]); y = np.array([r[1] for r in res])
    d = np.array([r[2] for r in res]); s = np.array([r[3] for r in res])
    nb = X.shape[1] - n_priv("pickmany")
    m = d < 0.04
    print(f"steps within 4 cm: {m.sum()}, teacher 'close' steps: {int(y[m].sum())}, trusses {a.n}")
    for name, sl in {"all brain inputs": slice(0, nb), "wrist camera only": slice(480, 960),
                     "privileged (teacher's)": slice(nb, None)}.items():
        Xs, cv = X[m][:, sl], GroupKFold(5)
        p = cross_val_predict(C(max_iter=300, learning_rate=0.05), Xs, y[m], groups=s[m], cv=cv, method="predict_proba")[:, 1]
        dh = cross_val_predict(R(max_iter=300, learning_rate=0.05), Xs, d[m], groups=s[m], cv=cv)
        err = np.abs(dh - d[m])
        print(f"{name:24s} close-moment AUC {roc_auc_score(y[m], p):.3f}  AP {average_precision_score(y[m], p):.3f} "
              f"(base {y[m].mean():.3f}) | distance: median error {1000 * np.median(err):.1f} mm, "
              f"90% {1000 * np.percentile(err, 90):.1f} mm")


if __name__ == "__main__":
    main()
