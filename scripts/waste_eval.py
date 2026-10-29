"""Waste-sorting station (panda_waste): the test battery for one trained controller (fly brain, control wiring or
MLP). The same items in every run (fixed seeds), the best brain, no retraining:

  clean      the trained item types, normal light
  heldout    two item types never shown in training (a white HDPE tub, a green painted can)
  smell_off  the material sensor (smell pathway) silent
  eyes_off   both compound eyes dark
  dim, bright   light x0.5 and x2

Per condition: solved (right bin), wrong bin, not placed, and sorting accuracy (right bin among items that
ended in a bin).

    python scripts/waste_eval.py RUN OUT.json [N=200]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

CONDITIONS = {"clean": {}, "heldout": {"item_types": "heldout"}, "smell_off": {"smell_off": True},
              "eyes_off": {"eyes_off": True}, "dim": {"light_choices": [0.5]}, "bright": {"light_choices": [2.0]}}
SEED0 = 70_000


def _chunk(args):
    run, kw, seeds = args
    import torch
    torch.set_num_threads(1)
    import evaluate as E
    model = E.load(run, best=True)
    rows = []
    for s in seeds:
        vn = E.make_norm(run, kw, int(s), best=True)
        obs = vn.reset()
        model.reset_memory()
        info, n = {}, 0
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, d, infos = vn.step(a)
            info, n = infos[0], n + 1
            if d[0]:
                break
        env = vn.venv.envs[0].unwrapped
        rows.append(dict(seed=int(s), item=info.get("item"), material=info.get("material"),
                         solved=bool(info.get("is_success")), wrong=bool(info.get("wrong_bin")),
                         in_bin=bool(info.get("in_bin")) or bool(info.get("wrong_bin")), steps=n,
                         held_ever=bool(getattr(env, "_held", False)) or bool(info.get("is_success"))))
        vn.close()
    return rows


def main():
    run, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    workers = int(os.environ.get("WORKERS", os.cpu_count() or 2))
    res = {"run": os.path.basename(os.path.normpath(run)), "n": n, "conditions": {}}
    t0 = time.time()
    for name, kw in CONDITIONS.items():
        seeds = np.arange(SEED0, SEED0 + n)
        jobs = [(run, kw, part) for part in np.array_split(seeds, workers)]
        with Pool(workers) as p:
            rows = sum(p.map(_chunk, jobs), [])
        solved = np.mean([r["solved"] for r in rows])
        wrong = np.mean([r["wrong"] for r in rows])
        binned = [r for r in rows if r["solved"] or r["wrong"]]
        acc = np.mean([r["solved"] for r in binned]) if binned else float("nan")
        per_item = {}
        for r in rows:
            per_item.setdefault(r["item"], []).append(r["solved"])
        res["conditions"][name] = dict(solved=float(solved), wrong_bin=float(wrong), not_placed=float(1 - solved - wrong),
                                       sort_accuracy=float(acc), median_steps=float(np.median([r["steps"] for r in rows if r["solved"]] or [0])),
                                       per_item={k: float(np.mean(v)) for k, v in sorted(per_item.items())})
        c = res["conditions"][name]
        print(f"{name:10s} solved {c['solved']:6.1%} | wrong bin {c['wrong_bin']:5.1%} | not placed {c['not_placed']:5.1%} | "
              f"sorting accuracy {c['sort_accuracy']:6.1%} | {time.time() - t0:.0f}s", flush=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
