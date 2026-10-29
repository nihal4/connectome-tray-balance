"""Store, in a fly-brain file, every descending neuron the brain contains that is not yet a command output.

The brain was built with 64 command neurons (the descending neurons with the most input), but the neurons kept
along the way include more descending neurons: 156 in data/flybrain_eye160_smell_N4476.npz. --all_dn
(scripts/train.py) reads all of them. Saved as `dn_extra_idx` (brain indices, sorted).

    python scripts/add_all_dn.py data/flybrain_eye160_smell_N4476.npz
"""
import os
import sys

import numpy as np
import pandas as pd


def main(path, raw="data/raw"):
    d = dict(np.load(path, allow_pickle=True))
    ann = pd.read_csv(os.path.join(raw, "neuron_annotations.tsv"), sep="\t", low_memory=False)
    con = pd.read_parquet(os.path.join(raw, "Connectivity_783.parquet"),
                          columns=["Presynaptic_ID", "Presynaptic_Index", "Postsynaptic_ID", "Postsynaptic_Index"])
    id2i = dict(zip(con.Presynaptic_ID, con.Presynaptic_Index))
    id2i.update(zip(con.Postsynaptic_ID, con.Postsynaptic_Index))
    ann["idx"] = ann.root_id.map(id2i)
    ann = ann.dropna(subset=["idx"]).astype({"idx": int}).drop_duplicates("idx").set_index("idx")
    sc = ann.reindex(d["root_ids"]).super_class.to_numpy()      # root_ids = connectome indices of the kept neurons
    dn = np.where(sc == "descending")[0]
    extra = np.setdiff1d(dn, d["output_idx"])
    assert np.isin(d["output_idx"], dn).all(), "the existing outputs should all be descending neurons"
    d["dn_extra_idx"] = extra.astype(np.int64)
    np.savez_compressed(path, **d)
    print(f"{path}: {len(dn)} descending neurons, {len(d['output_idx'])} already outputs, {len(extra)} added as dn_extra_idx")


if __name__ == "__main__":
    main(sys.argv[1])
