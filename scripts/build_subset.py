"""Cut a small sub-network out of the full fly connectome.

The whole brain (~139k neurons, ~15M connections) is too big for RL on a
laptop, so we keep the N most strongly connected neurons and every connection
between them (the "induced subgraph").

Inside that subset:
  input neurons  = neurons that mostly SEND (high out / in ratio)  -> receive robot sensors
  output neurons = neurons that mostly RECEIVE (high in / out ratio) -> read out as motor commands

Saves data/subset_N{N}.npz with:
  pre, post   edge endpoints (0..N-1)
  count       synapse count per edge
  sign        +1 excitatory / -1 inhibitory (kept fixed during training)
  input_idx, output_idx
  root_ids    original FlyWire neuron index of each subset neuron

Usage:
    python scripts/build_subset.py --n 1000
    python scripts/build_subset.py --synthetic --n 1000   # random graph, no download needed
"""
import argparse
import json
import os

import numpy as np


def from_flywire(path, n, n_in, n_out, min_syn):
    import pandas as pd

    cols = ["Presynaptic_Index", "Postsynaptic_Index", "Connectivity", "Excitatory"]
    df = pd.read_parquet(path, columns=cols)
    df = df[df.Connectivity >= min_syn]
    pre_all = df.Presynaptic_Index.to_numpy()
    post_all = df.Postsynaptic_Index.to_numpy()
    cnt_all = df.Connectivity.to_numpy()
    total = int(max(pre_all.max(), post_all.max()) + 1)

    strength = np.bincount(pre_all, cnt_all, total) + np.bincount(post_all, cnt_all, total)
    keep = np.argsort(-strength)[:n]
    remap = -np.ones(total, dtype=np.int64)
    remap[keep] = np.arange(n)
    m = (remap[pre_all] >= 0) & (remap[post_all] >= 0)
    pre = remap[pre_all[m]]
    post = remap[post_all[m]]
    count = cnt_all[m].astype(np.float32)
    sign = df.Excitatory.to_numpy()[m].astype(np.float32)
    return pre, post, count, sign, keep


def synthetic(n, density=0.05, seed=0):
    """Random sparse graph with the same format. Only for offline testing."""
    rng = np.random.default_rng(seed)
    e = int(n * n * density)
    pre = rng.integers(0, n, e)
    post = rng.integers(0, n, e)
    m = pre != post
    pre, post = pre[m], post[m]
    count = rng.geometric(0.4, len(pre)).astype(np.float32)
    sign = np.where(rng.random(n) < 0.6, 1.0, -1.0).astype(np.float32)[pre]
    return pre, post, count, sign, np.arange(n)


def pick_io(pre, post, count, n, n_in, n_out):
    out_s = np.bincount(pre, count, n)
    in_s = np.bincount(post, count, n)
    send_ratio = (out_s + 1) / (in_s + 1)
    input_idx = np.argsort(-send_ratio)[:n_in]
    recv_ratio = (in_s + 1) / (out_s + 1)
    recv_ratio[input_idx] = -np.inf
    output_idx = np.argsort(-recv_ratio)[:n_out]
    return np.sort(input_idx), np.sort(output_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/Connectivity_783.parquet")
    ap.add_argument("--n", type=int, default=1000, help="number of neurons to keep")
    ap.add_argument("--n_in", type=int, default=64)
    ap.add_argument("--n_out", type=int, default=64)
    ap.add_argument("--min_syn", type=int, default=1, help="drop edges with fewer synapses")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.synthetic:
        pre, post, count, sign, ids = synthetic(args.n)
        tag = "synthetic"
    else:
        if not os.path.exists(args.raw):
            raise SystemExit(f"{args.raw} not found. Run scripts/download_connectome.py first.")
        pre, post, count, sign, ids = from_flywire(args.raw, args.n, args.n_in, args.n_out, args.min_syn)
        tag = "flywire783"

    input_idx, output_idx = pick_io(pre, post, count, args.n, args.n_in, args.n_out)
    out = args.out or f"data/subset_{tag}_N{args.n}.npz"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, pre=pre.astype(np.int64), post=post.astype(np.int64), count=count,
                        sign=sign, input_idx=input_idx, output_idx=output_idx, root_ids=ids,
                        n=np.int64(args.n), source=tag)

    deg_in = np.bincount(post, minlength=args.n)
    stats = {
        "source": tag, "neurons": args.n, "edges": int(len(pre)),
        "density": round(len(pre) / args.n ** 2, 4),
        "excitatory_fraction": round(float((sign > 0).mean()), 3),
        "isolated_neurons": int(((deg_in == 0) & (np.bincount(pre, minlength=args.n) == 0)).sum()),
        "inputs": len(input_idx), "outputs": len(output_idx), "file": out,
    }
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
