"""Fly smell receptor responses for the simulated fruits, from the DoOR database.

DoOR (Database of Odor Responses; Muench & Galizia 2016, Sci Rep 6:21841; CC BY-SA 4.0) merges published
recordings of Drosophila olfactory receptor neurons into one consensus response per receptor and odorant,
scaled 0..1, with the spontaneous firing rate (SFR) as its own row. Here: response - SFR (so below 0 =
inhibited), for the receptor of each FlyWire glomerulus (ORN_<glomerulus> cell types), and for the
chemicals of the fruit recipes below. Unknown pairs (not measured) count as 0 and are reported.

Fruit recipes (relative amounts; simplified from the fruit-volatile and fermentation literature):
  ripe   fruity esters dominate (isopentyl acetate, ethyl butyrate, hexyl acetate, ethyl hexanoate),
         a little ethanol
  rotten fermentation products dominate (ethanol, acetic acid, ethyl acetate, diacetyl), some ester left

Usage:
    git clone --depth 1 https://github.com/ropensci/DoOR.data.git /tmp/DoOR.data
    python scripts/build_smell_table.py --door /tmp/DoOR.data/data
Writes data/smell_door.npz (+ the attribution in data/SMELL_DATA_NOTICE.md).
"""
import argparse
import os

import numpy as np
import pandas as pd

ODORANTS = {                      # name used here -> name in DoOR's odor.csv
    "isopentyl acetate": "isopentyl acetate",
    "ethyl butyrate": "ethyl butyrate",
    "hexyl acetate": "hexyl acetate",
    "ethyl hexanoate": "ethyl hexanoate",
    "ethanol": "ethanol",
    "acetic acid": "acetic acid",
    "ethyl acetate": "ethyl acetate",
    "diacetyl": "2,3-butanedione",
}
RECIPES = {
    "ripe": {"isopentyl acetate": 1.0, "ethyl butyrate": 0.6, "hexyl acetate": 0.5, "ethyl hexanoate": 0.4,
             "ethanol": 0.1},
    "rotten": {"ethanol": 1.0, "acetic acid": 0.8, "ethyl acetate": 0.7, "diacetyl": 0.3,
               "isopentyl acetate": 0.2},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--door", required=True, help="DoOR.data/data folder")
    ap.add_argument("--ann", default="data/raw/neuron_annotations.tsv")
    ap.add_argument("--out", default="data/smell_door.npz")
    args = ap.parse_args()
    rm = pd.read_csv(os.path.join(args.door, "door_response_matrix.csv"), sep=";", index_col=0)
    mp = pd.read_csv(os.path.join(args.door, "door_mappings.csv"), sep=";")
    od = pd.read_csv(os.path.join(args.door, "odor.csv"), sep=";")
    ann = pd.read_csv(args.ann, sep="\t", usecols=["cell_type"], low_memory=False)
    glos = sorted(t[4:] for t in ann.cell_type.dropna().unique() if t.startswith("ORN_"))
    sfr = rm.loc["SFR"]
    keys = {}
    for name, door_name in ODORANTS.items():
        hit = od[od.Name.astype(str).str.lower() == door_name]
        assert len(hit) and hit.InChIKey.iloc[0] in rm.index, name
        keys[name] = hit.InChIKey.iloc[0]
    resp = np.zeros((len(glos), len(ODORANTS)), np.float32)
    measured = np.zeros_like(resp, bool)
    receptor = []
    for gi, g in enumerate(glos):
        rows = mp[(mp.code == g) | (mp.glomerulus == g)]
        recs = [r for r in rows.receptor.astype(str) if r in rm.columns]
        recs = sorted(recs, key=lambda r: not r.startswith(("Or", "Ir", "Gr")))   # named receptor first
        receptor.append(recs[0] if recs else "")
        if not recs:
            continue
        for oi, name in enumerate(ODORANTS):
            v = rm.loc[keys[name], recs[0]]
            if pd.notna(v):
                resp[gi, oi] = float(v) - float(sfr[recs[0]])
                measured[gi, oi] = True
    names = list(ODORANTS)
    rec = np.array([[RECIPES[f].get(n, 0.0) for n in names] for f in ("ripe", "rotten")], np.float32)
    np.savez(args.out, glomeruli=np.array(glos), receptor=np.array(receptor), odorants=np.array(names),
             response=resp, measured=measured, fruits=np.array(["ripe", "rotten"]), recipes=rec)
    with open(os.path.join(os.path.dirname(args.out), "SMELL_DATA_NOTICE.md"), "w") as f:
        f.write("smell_door.npz is derived from DoOR.data (https://github.com/ropensci/DoOR.data), "
                "Muench D, Galizia CG (2016) DoOR 2.0 - Comprehensive Mapping of Drosophila melanogaster Odorant "
                "Responses. Sci Rep 6:21841. Licensed CC BY-SA 4.0; the derived table is shared under the same "
                "licence. Built by scripts/build_smell_table.py.\n")
    print(f"{len(glos)} glomeruli, {sum(r != '' for r in receptor)} with DoOR data; "
          f"{measured.sum()} of {measured.size} glomerulus x odorant pairs measured")
    drive = rec @ resp.T                                       # (fruit, glomerulus)
    for fi, fr in enumerate(("ripe", "rotten")):
        top = np.argsort(-drive[fi])[:6]
        print(f"  {fr:6s} strongest glomeruli: " + ", ".join(f"{glos[i]} {drive[fi, i]:.2f}" for i in top))
    d = drive[0] - drive[1]
    print("  ripe minus rotten, most different: " + ", ".join(f"{glos[i]} {d[i]:+.2f}" for i in np.argsort(-np.abs(d))[:8]))


if __name__ == "__main__":
    main()
