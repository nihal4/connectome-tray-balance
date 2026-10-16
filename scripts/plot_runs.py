"""Plot eval curves of several runs (same environment) on one chart.

    python scripts/plot_runs.py runs/maze3-connectome-real-s0 runs/maze3-mlp-s0 --out curves.png

Panels: eval return, episode length, and success rate (maze runs).
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

    data = {}
    for run in args.runs:
        f = os.path.join(run, "evaluations.npz")
        if not os.path.exists(f):
            print(f"skip {run}: no evaluations.npz yet")
            continue
        data[os.path.basename(run.rstrip("/"))] = dict(np.load(f))
    has_success = any("successes" in d and d["successes"].size for d in data.values())

    fig, axes = plt.subplots(1, 3 if has_success else 2, figsize=(15 if has_success else 11, 4))
    for label, d in data.items():
        axes[0].plot(d["timesteps"], d["results"].mean(1), label=label)
        axes[1].plot(d["timesteps"], d["ep_lengths"].mean(1), label=label)
        if has_success and "successes" in d and d["successes"].size:
            axes[2].plot(d["timesteps"], d["successes"].mean(1) * 100, label=label)
    axes[0].set(title="Eval return", xlabel="env steps", ylabel="return")
    axes[1].set(title="Episode length", xlabel="env steps", ylabel="steps")
    if has_success:
        axes[2].set(title="Solved (%)", xlabel="env steps", ylabel="% of eval episodes", ylim=(-5, 105))
    from matplotlib.ticker import FuncFormatter
    for ax in axes:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:g}M" if x >= 1e6 else f"{x / 1e3:g}k"))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
