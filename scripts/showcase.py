"""Show what a brain's successes look like.

The evaluation's own episodes are run without a camera (rendering would slow training), so they
are not saved. But the brain is deterministic: the same layout always gives the same result. So
this runs the BEST brain on N fresh random layouts with the camera off, notes which ones it
solves, and then replays up to K of them with the camera on (identical episodes), plus one miss
for contrast. Same success rule as the evaluation (solved at the end of the episode).

    python scripts/showcase.py --run ~/flybrain/runs/<run> --episodes 40 --videos 3
Output: runs/<run>/progress/showcase_<steps>k.mp4 (shown on the dashboard).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--videos", type=int, default=3, help="solved episodes to film")
    ap.add_argument("--first_seed", type=int, default=50_000)
    args = ap.parse_args()
    import numpy as np
    from ctb.envs import build_env
    from scripts.evaluate import load, make_norm, run_config, run_env_kw
    from scripts.desktop.watch import record_tests
    run = os.path.abspath(os.path.expanduser(args.run))
    model = load(run, best=True)
    vn = make_norm(run, {})
    env_name = run_config(run)["env"]
    solved, missed = [], []
    for i in range(args.episodes):
        seed = args.first_seed + i
        env = build_env(env_name, seed, **run_env_kw(run))
        obs, _ = env.reset(seed=seed)
        info = {}
        for _ in range(env.max_steps):
            a, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
            obs, _, te, tr, info = env.step(a[0])
            if te or tr:
                break
        (solved if info.get("is_success") else missed).append(seed)
        env.close()
    print(f"best brain on {args.episodes} new random layouts: solved {len(solved)} "
          f"({100 * len(solved) / args.episodes:.0f} %)")
    if not solved:
        print("no solved episode to show")
        return
    steps = 0
    try:
        steps = int(np.load(os.path.join(run, "evaluations.npz"))["timesteps"][-1])
    except Exception:
        pass
    out = os.path.join(run, "progress", f"showcase_{steps // 1000:05d}k.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    seeds = solved[:args.videos] + missed[:1]
    record_tests(run, out, seeds=seeds, best=True)
    print(f"video: {out}  ({min(len(solved), args.videos)} solved + {min(len(missed), 1)} missed, best brain)")


if __name__ == "__main__":
    main()
