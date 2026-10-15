"""Plot eval curves of several runs on one chart.

    python scripts/plot_runs.py runs/connectome-real-s0 runs/mlp-s0 --out curves.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="curves.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run in args.runs:
        f = os.path.join(run, "evaluations.npz")
        if not os.path.exists(f):
            print(f"skip {run}: no evaluations.npz yet")
            continue
        d = np.load(f)
        label = os.path.basename(run.rstrip("/"))
        axes[0].plot(d["timesteps"], d["results"].mean(1), label=label)
        axes[1].plot(d["timesteps"], d["ep_lengths"].mean(1), label=label)
    axes[0].set(title="Eval return", xlabel="env steps", ylabel="return")
    axes[1].set(title="Steps survived (max 500)", xlabel="env steps", ylabel="episode length")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
