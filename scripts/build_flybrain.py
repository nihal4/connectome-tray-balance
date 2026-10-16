"""Build the "fly-robot" brain: a sub-network of the real fly brain wired from the
fly's own sense organs to its own motor-command neurons.

    robot camera  ->  photoreceptors of the right eye (R8 = this frame, R7 = previous frame)
    robot joints  ->  ascending neurons (body sense coming up from the nerve cord)
    ...           ->  neurons on the shortest pathways (optic lobe, central brain)
    arm commands  <-  descending neurons (the brain's commands to the body)

Pathway selection, layer by layer: starting from the input neurons, each layer adds
the neurons that receive the most synapses from the layer before AND can still reach
a descending neuron in the hops left. Then every connection among the chosen neurons
is kept (so loops and lateral wiring are real, too).

Pixels are mapped to photoreceptors roughly retinotopically: neurons are sorted into a
grid by their position in the eye, so neighbouring pixels go to neighbouring receptors.
(Approximate: positions are neuron coordinates, not exact eye columns.)

Usage:
    python scripts/build_flybrain.py                 # 16x16 camera, ~2500 neurons
    python scripts/build_flybrain.py --n 3000 --hops 4

Needs data/raw/Connectivity_783.parquet and data/raw/neuron_annotations.tsv
(scripts/download_connectome.py gets both).
"""
import argparse
import collections
import json
import os

import numpy as np
import pandas as pd


def grid_order(df, side):
    """Pick side*side neurons spread over the eye and order them row by row."""
    df = df.sort_values("pos_y").reset_index(drop=True)
    rows = []
    for ix in np.array_split(np.arange(len(df)), side):
        band = df.iloc[ix].sort_values("pos_z")
        pick = np.linspace(0, len(band) - 1, side).round().astype(int)
        rows.append(band.iloc[pick])
    return pd.concat(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--n", type=int, default=2500, help="total neurons")
    ap.add_argument("--cam", type=int, default=16, help="camera is cam x cam pixels")
    ap.add_argument("--n_asc", type=int, default=64)
    ap.add_argument("--n_out", type=int, default=64)
    ap.add_argument("--hops", type=int, default=4, help="layers between eye and commands")
    ap.add_argument("--min_syn", type=int, default=5, help="ignore connections weaker than this")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ann = pd.read_csv(os.path.join(args.raw, "neuron_annotations.tsv"), sep="\t", low_memory=False)
    con = pd.read_parquet(os.path.join(args.raw, "Connectivity_783.parquet"),
                          columns=["Presynaptic_ID", "Postsynaptic_ID", "Presynaptic_Index",
                                   "Postsynaptic_Index", "Connectivity", "Excitatory"])
    id2i = dict(zip(con.Presynaptic_ID, con.Presynaptic_Index))
    id2i.update(zip(con.Postsynaptic_ID, con.Postsynaptic_Index))
    ann["idx"] = ann.root_id.map(id2i)
    ann = ann.dropna(subset=["idx"]).astype({"idx": int})
    con = con[con.Connectivity >= args.min_syn]
    total = int(max(con.Presynaptic_Index.max(), con.Postsynaptic_Index.max()) + 1)
    pre_all, post_all = con.Presynaptic_Index.to_numpy(), con.Postsynaptic_Index.to_numpy()
    cnt_all = con.Connectivity.to_numpy().astype(np.float32)
    out_str = np.bincount(pre_all, cnt_all, total)
    in_str = np.bincount(post_all, cnt_all, total)

    # ---- inputs and outputs, by real cell type
    eye = ann[(ann.side == "right") & (ann.cell_type.isin(["R7", "R8"]))]
    eye = eye[out_str[eye.idx] > 0]
    npx = args.cam * args.cam
    r8 = grid_order(eye[eye.cell_type == "R8"], args.cam).idx.to_numpy()   # current frame
    r7 = grid_order(eye[eye.cell_type == "R7"], args.cam).idx.to_numpy()   # previous frame
    assert len(r8) == npx and len(r7) == npx, "not enough photoreceptors for this camera size"
    vis = np.concatenate([r8, r7])

    asc_all = ann[ann.super_class == "ascending"].idx.to_numpy()
    asc = asc_all[np.argsort(-out_str[asc_all])][:args.n_asc]
    dn_all = ann[ann.super_class == "descending"].idx.to_numpy()
    dn = dn_all[np.argsort(-in_str[dn_all])][:args.n_out]

    # ---- hops-to-output for every neuron (backward BFS from descending neurons)
    radj = collections.defaultdict(list)
    adj = collections.defaultdict(list)
    for p, q, w in zip(pre_all, post_all, cnt_all):
        adj[p].append((q, w))
        radj[q].append(p)
    to_out = {d: 0 for d in dn}
    dq = collections.deque(dn)
    while dq:
        u = dq.popleft()
        if to_out[u] >= args.hops + 1:
            continue
        for v in radj[u]:
            if v not in to_out:
                to_out[v] = to_out[u] + 1
                dq.append(v)

    # ---- layer-by-layer pathway selection, separately for the eye stream (most of the
    # budget) and the body-sense stream; otherwise the strong ascending inputs win every
    # slot and the weaker visual pathway never reaches the command neurons.
    io = set(vis) | set(asc) | set(dn)
    budget = args.n - len(io)
    chosen = set(io)
    layers = {}

    def grow(sources, n_total, name):
        per_layer = n_total // args.hops
        frontier = set(sources)
        sizes = []
        for h in range(1, args.hops + 1):
            score = collections.defaultdict(float)
            for u in frontier:
                for v, w in adj[u]:
                    if v not in chosen and to_out.get(v, 99) <= args.hops + 1 - h:
                        score[v] += w
            best = sorted(score, key=score.get, reverse=True)[:per_layer]
            sizes.append(len(best))
            chosen.update(best)
            frontier = set(best)
        layers[name] = sizes

    grow(vis, int(budget * 0.8), "eye_stream")
    grow(asc, budget - int(budget * 0.8), "body_stream")
    keep = np.array(sorted(chosen))

    remap = -np.ones(total, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    m = (remap[pre_all] >= 0) & (remap[post_all] >= 0)
    pre, post = remap[pre_all[m]], remap[post_all[m]]
    count = cnt_all[m]
    sign = con.Excitatory.to_numpy()[m].astype(np.float32)

    # ---- check: how many hops from eye to command neurons inside the subset
    sadj = collections.defaultdict(list)
    for p, q in zip(pre, post):
        sadj[p].append(q)
    dist = {v: 0 for v in remap[vis]}
    dq = collections.deque(dist)
    while dq:
        u = dq.popleft()
        for v in sadj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
    hops = [dist.get(d, -1) for d in remap[dn]]

    sc = ann.set_index("idx").reindex(keep).super_class.fillna("unknown")
    n = len(keep)
    out = args.out or f"data/flybrain_cam{args.cam}_N{n}.npz"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(
        out, pre=pre, post=post, count=count, sign=sign, n=np.int64(n), source="flywire783-flyrobot",
        vis_idx=remap[vis], asc_idx=remap[asc], output_idx=remap[dn],
        input_idx=np.concatenate([remap[vis], remap[asc]]), root_ids=keep, cam=np.int64(args.cam))
    stats = {
        "file": out, "neurons": n, "connections": int(len(pre)),
        "eye_inputs": int(len(vis)), "body_sense_inputs": int(len(asc)), "command_outputs": int(len(dn)),
        "pathway_layers": layers,
        "brain_parts_%": (sc.value_counts(normalize=True) * 100).round(1).to_dict(),
        "eye_to_command_hops": dict(sorted(collections.Counter(hops).items())),
    }
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
