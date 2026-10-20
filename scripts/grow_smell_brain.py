"""Grow the fly-robot brain by the fly's smell pathway, keeping everything it already has.

The existing brain (eyes, body sense, command neurons, the pathways between them) stays exactly as it
is: same neurons in the same order, same synapses in the same order, so a trained brain's connection
strengths carry over one to one. Appended: one olfactory receptor neuron (ORN) per glomerulus type and
antenna (FlyWire ORN_<glomerulus> types; left antenna -> left ORNs), and the neurons on the strongest
pathways from them into the existing brain, chosen layer by layer like the eye pathways
(build_flybrain.py): every ORN keeps a route (its strongest partner one synapse closer), the rest of each
layer is filled with the most strongly driven neurons. Then every synapse between a new neuron and any
neuron of the grown brain is added (real wiring, sign from the neurotransmitter, as before).

Usage:
    python scripts/grow_smell_brain.py --base data/flybrain_eye160_N4000.npz --smell data/smell_door.npz \
        --n_new 600 --hops 4
"""
import argparse
import collections
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--base", default="data/flybrain_eye160_N4000.npz")
    ap.add_argument("--smell", default="data/smell_door.npz")
    ap.add_argument("--n_new", type=int, default=600, help="neurons to add (incl. the receptor neurons)")
    ap.add_argument("--hops", type=int, default=4, help="layers from receptor neurons into the brain")
    ap.add_argument("--min_syn", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = np.load(args.base)
    smell = np.load(args.smell)
    glos = list(smell["glomeruli"])
    old = base["root_ids"].astype(np.int64)            # connectome indices of the existing neurons
    n_old = len(old)
    old_set = set(old.tolist())

    ann = pd.read_csv(os.path.join(args.raw, "neuron_annotations.tsv"), sep="\t", low_memory=False)
    con = pd.read_parquet(os.path.join(args.raw, "Connectivity_783.parquet"),
                          columns=["Presynaptic_ID", "Postsynaptic_ID", "Presynaptic_Index",
                                   "Postsynaptic_Index", "Connectivity", "Excitatory"])
    id2i = dict(zip(con.Presynaptic_ID, con.Presynaptic_Index))
    id2i.update(zip(con.Postsynaptic_ID, con.Postsynaptic_Index))
    ann["idx"] = ann.root_id.map(id2i)
    ann = ann.dropna(subset=["idx"]).astype({"idx": int})
    con = con[con.Connectivity >= args.min_syn]
    pre_all, post_all = con.Presynaptic_Index.to_numpy(), con.Postsynaptic_Index.to_numpy()
    cnt_all = con.Connectivity.to_numpy().astype(np.float32)
    exc_all = con.Excitatory.to_numpy().astype(np.float32)
    total = int(max(pre_all.max(), post_all.max(), ann.idx.max()) + 1)
    out_str = np.bincount(pre_all, cnt_all, total)
    adj = collections.defaultdict(list)
    radj = collections.defaultdict(list)
    for p, q, w in zip(pre_all, post_all, cnt_all):
        adj[p].append((q, w))
        radj[q].append(p)

    # distance (synapses) from every neuron to the existing brain
    to_old = {v: 0 for v in old_set}
    dq = collections.deque(to_old)
    while dq:
        u = dq.popleft()
        if to_old[u] >= args.hops:
            continue
        for v in radj[u]:
            if v not in to_old:
                to_old[v] = to_old[u] + 1
                dq.append(v)

    # inputs: per antenna (left, right) and glomerulus, the ORN with the strongest output that can reach
    # the existing brain; a missing glomerulus/side is skipped (and recorded)
    smell_idx, smell_glo, smell_side = [], [], []
    for s, side in enumerate(("left", "right")):
        for gi, g in enumerate(glos):
            c = ann[(ann.cell_type == f"ORN_{g}") & (ann.side == side)].idx.to_numpy()
            c = [i for i in c if to_old.get(i, 99) <= args.hops and i not in old_set]
            if not c:
                continue
            smell_idx.append(max(c, key=lambda i: out_str[i]))
            smell_glo.append(gi)
            smell_side.append(s)
    orn = list(dict.fromkeys(smell_idx))
    print(f"receptor neurons: {len(orn)} ({len(glos)} glomerulus types x 2 antennae possible)")

    # layer-by-layer pathway selection from the receptor neurons towards the existing brain
    chosen = set(orn)
    left = args.n_new - len(orn)
    frontier = set(orn)
    for h in range(1, args.hops + 1):
        per_layer = left // (args.hops + 1 - h)
        route = set()
        for u in frontier:                              # every neuron keeps its best step inwards
            k = to_old.get(u, 99)
            best = [(w, v) for v, w in adj[u] if to_old.get(v, 99) == k - 1]
            if best:
                route.add(max(best)[1])
        new = [v for v in route if v not in chosen and v not in old_set]
        score = collections.defaultdict(float)
        for u in frontier:
            for v, w in adj[u]:
                if v not in chosen and v not in old_set and v not in route and to_old.get(v, 99) <= args.hops - h:
                    score[v] += w
        fill = sorted(score, key=score.get, reverse=True)[:max(per_layer - len(new), 0)]
        chosen.update(new)
        chosen.update(fill)
        left -= len(new) + len(fill)
        frontier = (route | set(fill)) - old_set
        print(f"  layer {h}: +{len(new) + len(fill)} neurons (routes {len(new)}, strongest {len(fill)})")
    new_ids = np.array(sorted(chosen), dtype=np.int64)
    keep = np.concatenate([old, new_ids])
    n = len(keep)
    remap = -np.ones(total, dtype=np.int64)
    remap[keep] = np.arange(n)

    # synapses: the old ones unchanged and in the same order, then every synapse touching a new neuron
    m_new = (remap[pre_all] >= 0) & (remap[post_all] >= 0) & ((remap[pre_all] >= n_old) | (remap[post_all] >= n_old))
    pre = np.concatenate([base["pre"], remap[pre_all[m_new]]])
    post = np.concatenate([base["post"], remap[post_all[m_new]]])
    count = np.concatenate([base["count"], cnt_all[m_new]])
    sign = np.concatenate([base["sign"], exc_all[m_new]])

    # check: receptor neurons -> command neurons inside the grown brain
    sadj = collections.defaultdict(list)
    for p, q in zip(pre, post):
        sadj[p].append(q)
    dist = {v: 0 for v in remap[orn]}
    dq = collections.deque(dist)
    while dq:
        u = dq.popleft()
        for v in sadj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
    h_dn = [dist.get(d, -1) for d in base["output_idx"]]
    reached = [h for h in h_dn if h >= 0]
    sc = ann.set_index("idx").reindex(new_ids).cell_class.fillna("unknown").value_counts()
    print(f"grown brain: {n} neurons ({n_old} + {len(new_ids)}), {len(pre)} synapses ({len(base['pre'])} + {m_new.sum()})")
    print(f"command neurons reached from the receptor neurons: {len(reached)} of {len(h_dn)}, "
          f"median {np.median(reached) if reached else '-'} synapses")
    print("new neurons by class:", ", ".join(f"{k} {v}" for k, v in sc.head(10).items()))
    out = args.out or f"data/flybrain_eye{int(base['facets'])}_smell_N{n}.npz"
    extra = {k: base[k] for k in base.files if k not in ("pre", "post", "count", "sign", "n", "root_ids", "input_idx", "source")}
    np.savez_compressed(out, pre=pre, post=post, count=count, sign=sign, n=np.int64(n),
                        source="flywire783-flyrobot+smell", root_ids=keep,
                        input_idx=np.concatenate([base["input_idx"], remap[smell_idx]]),
                        smell_idx=remap[smell_idx], smell_glo=np.array(smell_glo), smell_side=np.array(smell_side),
                        smell_glomeruli=np.array(glos), grown_from=os.path.basename(args.base),
                        n_old=np.int64(n_old), e_old=np.int64(len(base["pre"])), **extra)
    print("saved", out)


if __name__ == "__main__":
    main()
