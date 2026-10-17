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

Compound-eye version (fly-brain drone, ctb/envs/drone.py): --eye compound
    both eyes; every facet drives its own R1-6 (brightness), R7 (UV) and R8 (green)
    photoreceptor, chosen so that the facet layout matches the photoreceptor layout
    (rough retinotopy from neuron positions, matched with an optimal assignment)

Usage:
    python scripts/build_flybrain.py                 # 16x16 camera, ~2500 neurons
    python scripts/build_flybrain.py --n 3000 --hops 4
    python scripts/build_flybrain.py --eye compound --facets 160 --n 3000 --hops 5

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


def sheet_coords(df):
    """2D position of neurons on the eye sheet: elevation from the dorso-ventral axis
    (FlyWire y grows ventrally) and azimuth from the main remaining axis. Rank-normalised
    to [0, 1]. Approximate: neuron coordinates, not exact eye columns."""
    xyz = df[["pos_x", "pos_y", "pos_z"]].to_numpy(float)
    xyz = xyz - xyz.mean(0)
    _, _, vt = np.linalg.svd(xyz, full_matrices=False)
    pcs = xyz @ vt[:2].T
    k = int(np.argmax(np.abs(np.corrcoef(np.c_[pcs, xyz[:, 1]].T)[2, :2])))   # PC closest to y
    el = -pcs[:, k] * np.sign(np.corrcoef(pcs[:, k], xyz[:, 1])[0, 1])         # dorsal = up
    az = pcs[:, 1 - k]
    rank = lambda a: np.argsort(np.argsort(a)) / (len(a) - 1)  # noqa: E731
    return np.c_[rank(az), rank(el)]


def match_facets(df, facet_uv):
    """Pick one neuron per facet so that neighbouring facets get neighbouring neurons."""
    from scipy.optimize import linear_sum_assignment
    xy = sheet_coords(df)
    cost = ((facet_uv[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    rows, cols = linear_sum_assignment(cost)
    return df.idx.to_numpy()[cols[np.argsort(rows)]]


def compound_eye(ann, out_str, n_facets):
    """vis neurons in the drone's observation order: right eye then left eye, per facet
    (R1-6, R7, R8)."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ctb.envs.drone import eye_facets
    _, az, el, _ = eye_facets(n_facets)
    rank = lambda a: np.argsort(np.argsort(a)) / (len(a) - 1)  # noqa: E731
    vis = []
    for side in ("right", "left"):
        a = -az if side == "right" else az            # front (midline) -> back, same for both eyes
        uv = np.c_[rank(a), rank(el)]
        per_type = []
        for ct in ("R1-6", "R7", "R8"):
            cells = ann[(ann.side == side) & (ann.cell_type == ct)]
            cells = cells[out_str[cells.idx] > 0]
            per_type.append(match_facets(cells, uv))
        vis.append(np.stack(per_type, 1).ravel())      # facet-major: R1-6, R7, R8
    return np.concatenate(vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--n", type=int, default=2500, help="total neurons")
    ap.add_argument("--cam", type=int, default=16, help="camera is cam x cam pixels")
    ap.add_argument("--eye", choices=["cam", "compound"], default="cam")
    ap.add_argument("--facets", type=int, default=160, help="facets per eye (--eye compound)")
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
    total = int(max(con.Presynaptic_Index.max(), con.Postsynaptic_Index.max(), ann.idx.max()) + 1)
    pre_all, post_all = con.Presynaptic_Index.to_numpy(), con.Postsynaptic_Index.to_numpy()
    cnt_all = con.Connectivity.to_numpy().astype(np.float32)
    out_str = np.bincount(pre_all, cnt_all, total)
    in_str = np.bincount(post_all, cnt_all, total)

    radj = collections.defaultdict(list)
    adj = collections.defaultdict(list)
    for p, q, w in zip(pre_all, post_all, cnt_all):
        adj[p].append((q, w))
        radj[q].append(p)

    def hops_to(targets, graph, limit):
        """Breadth-first distance (in synapses) to/from a set of neurons, up to limit."""
        dist = {t: 0 for t in targets}
        dq = collections.deque(dist)
        while dq:
            u = dq.popleft()
            if dist[u] >= limit:
                continue
            for v in graph[u]:
                v = v[0] if isinstance(v, tuple) else v
                if v not in dist:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        return dist

    asc_all = ann[ann.super_class == "ascending"].idx.to_numpy()
    asc = asc_all[np.argsort(-out_str[asc_all])][:args.n_asc]
    dn_ann = ann[ann.super_class == "descending"]

    # ---- inputs and outputs, by real cell type
    if args.eye == "compound":
        # Many photoreceptors are dead ends in the reconstruction (no path to any command
        # neuron), and the strongest command neurons are not all visual. So: keep only
        # photoreceptors with a path to a command neuron, and pick command neurons that
        # the eyes can reach, half on each side of the brain.
        to_any = hops_to(dn_ann.idx.to_numpy(), radj, args.hops + 1)
        pr = ann[ann.cell_type.isin(["R1-6", "R7", "R8"]) & ann.side.isin(["left", "right"])]
        pr = pr[[to_any.get(i, 99) <= args.hops for i in pr.idx]]
        from_eye = {side: hops_to(pr[pr.side == side].idx.to_numpy(), adj, args.hops)
                    for side in ("left", "right")}
        dn = []
        for side in ("left", "right"):
            cand = dn_ann[dn_ann.side == side].idx.to_numpy()
            both = np.array([(i in from_eye["left"]) + (i in from_eye["right"]) for i in cand])
            order = np.lexsort((-in_str[cand], -both))          # seen by both eyes first
            dn += list(cand[order][:args.n_out // 2])
        dn = np.array(dn)
        to_out = hops_to(dn, radj, args.hops + 1)
        pr = pr[[to_out.get(i, 99) <= args.hops for i in pr.idx]]
        vis = compound_eye(pr, out_str, args.facets)
    else:
        eye = ann[(ann.side == "right") & (ann.cell_type.isin(["R7", "R8"]))]
        eye = eye[out_str[eye.idx] > 0]
        npx = args.cam * args.cam
        r8 = grid_order(eye[eye.cell_type == "R8"], args.cam).idx.to_numpy()   # current frame
        r7 = grid_order(eye[eye.cell_type == "R7"], args.cam).idx.to_numpy()   # previous frame
        assert len(r8) == npx and len(r7) == npx, "not enough photoreceptors for this camera size"
        vis = np.concatenate([r8, r7])
        dn_all = dn_ann.idx.to_numpy()
        dn = dn_all[np.argsort(-in_str[dn_all])][:args.n_out]
        to_out = hops_to(dn, radj, args.hops + 1)

    # ---- layer-by-layer pathway selection, separately for the eye stream (most of the
    # budget) and the body-sense stream; otherwise the strong ascending inputs win every
    # slot and the weaker visual pathway never reaches the command neurons.
    io = set(vis) | set(asc) | set(dn)
    budget = args.n - len(io)
    chosen = set(io)
    layers = {}

    def grow(sources, n_total, name, cover=False):
        """cover=True (compound eye): first keep, for every neuron of the layer before, its
        strongest partner one step closer to the commands, so every facet keeps a route;
        then fill the rest of the layer with the most strongly driven neurons."""
        left = n_total
        frontier = set(sources)
        sizes = []
        for h in range(1, args.hops + 1):
            per_layer = left // (args.hops + 1 - h) if cover else n_total // args.hops
            nxt = set()
            if cover:
                for u in frontier:
                    k = to_out.get(u, 99)
                    best = [(w, v) for v, w in adj[u] if to_out.get(v, 99) == k - 1]
                    if best:
                        nxt.add(max(best)[1])
            new = [v for v in nxt if v not in chosen]
            score = collections.defaultdict(float)
            for u in frontier:
                for v, w in adj[u]:
                    if v not in chosen and v not in nxt and to_out.get(v, 99) <= args.hops + 1 - h:
                        score[v] += w
            fill = sorted(score, key=score.get, reverse=True)[:max(per_layer - len(new), 0)]
            chosen.update(new)
            chosen.update(fill)
            sizes.append(len(new) + len(fill))
            left -= len(new) + len(fill)
            frontier = (nxt | set(fill)) - set(dn) if cover else set(fill)
        layers[name] = sizes

    if args.eye == "compound":
        # body sense first with a fixed 20 % share (the per-facet routes would use up everything)
        grow(asc, budget - int(budget * 0.8), "body_stream")
        grow(vis, max(args.n - len(chosen), 0), "eye_stream", cover=True)
    else:                                   # unchanged order: same camera brain as before
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
    tag = f"eye{args.facets}" if args.eye == "compound" else f"cam{args.cam}"
    out = args.out or f"data/flybrain_{tag}_N{n}.npz"
    extra = dict(vis_dyn=np.int64(1), facets=np.int64(args.facets)) if args.eye == "compound" else dict(cam=np.int64(args.cam))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(
        out, pre=pre, post=post, count=count, sign=sign, n=np.int64(n), source="flywire783-flyrobot",
        vis_idx=remap[vis], asc_idx=remap[asc], output_idx=remap[dn],
        input_idx=np.concatenate([remap[vis], remap[asc]]), root_ids=keep, **extra)
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
