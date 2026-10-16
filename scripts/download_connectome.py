"""Download the FlyWire whole-brain connectome (v783) edge list.

Source: the public GitHub repo of Shiu et al. 2024 (Nature),
"A Drosophila computational brain model reveals sensorimotor processing",
https://github.com/philshiu/Drosophila_brain_model (MIT license).
Their file is derived from FlyWire (Dorkenwald et al. 2024, Nature).
Please cite both if you use it.

File: Connectivity_783.parquet (~100 MB, ~15M rows)
Columns: Presynaptic_Index, Postsynaptic_Index, Connectivity (synapse count),
         Excitatory (+1 / -1, from predicted neurotransmitter), ...

Also downloads the FlyWire cell-type annotations (~32 MB) from
https://github.com/flyconnectome/flywire_annotations (Schlegel et al. 2024).

Usage:
    python scripts/download_connectome.py --out data/raw
"""
import argparse
import os
import sys
import time
import urllib.request

BASE = "https://github.com/philshiu/Drosophila_brain_model/raw/main/"
FILES = {
    "Connectivity_783.parquet": 90_000_000,   # minimum expected size in bytes
    "Completeness_783.csv": 3_000_000,
}
# Cell-type annotations (Schlegel et al. 2024, Nature), same v783 neuron IDs.
# Needed for the fly-robot brain (eye, body-sense and command neurons).
ANNOTATIONS = ("https://github.com/flyconnectome/flywire_annotations/raw/main/supplemental_files/"
               "Supplemental_file1_neuron_annotations.tsv", "neuron_annotations.tsv", 30_000_000)


def download(url: str, dest: str, min_size: int):
    if os.path.exists(dest) and os.path.getsize(dest) >= min_size:
        print(f"[skip] {dest} already exists ({os.path.getsize(dest) / 1e6:.1f} MB)")
        return
    tmp = dest + ".part"
    print(f"[get ] {url}")
    t0 = time.time()

    def hook(blocks, bs, total):
        done = blocks * bs
        if total > 0:
            pct = min(100, done * 100 / total)
            sys.stdout.write(f"\r       {done / 1e6:7.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, hook)
    print()
    size = os.path.getsize(tmp)
    if size < min_size:
        os.remove(tmp)
        raise RuntimeError(
            f"Downloaded file is only {size} bytes (expected >= {min_size}). "
            "GitHub may have served a Git LFS pointer or an error page. "
            "Download it manually from the repo page and put it in the output folder."
        )
    os.replace(tmp, dest)
    print(f"[done] {dest} ({size / 1e6:.1f} MB in {time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, min_size in FILES.items():
        download(BASE + name, os.path.join(args.out, name), min_size)
    url, name, min_size = ANNOTATIONS
    download(url, os.path.join(args.out, name), min_size)


if __name__ == "__main__":
    main()
