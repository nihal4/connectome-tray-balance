"""Progress videos + dashboard for the training PC (runs as the flybrain-watch service).

Every minute: for each run in ~/flybrain/runs, when training has passed another 100k steps,
fly one test episode with the newest checkpoint (always the same room / maze, so the videos
can be compared) and save runs/<run>/progress/step_<N>k.mp4. Then rewrite ~/flybrain/index.html
(learning curves, latest numbers, CPU temperature, all progress videos), which the
flybrain-web service shows at http://<pc>:8080.

Runs at the lowest priority with 1 thread, so it barely slows training.
"""
import argparse
import glob
import html
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_REPO)

from ctb.safety import cpu_temp_c, limit_threads  # noqa: E402

limit_threads(1)

import numpy as np  # noqa: E402


def last_eval(run):
    f = os.path.join(run, "evaluations.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f)
    if not len(d["timesteps"]):
        return None
    out = {"steps": int(d["timesteps"][-1]), "return": float(d["results"][-1].mean()),
           "best_return": float(d["results"].mean(1).max())}
    if "successes" in d.files and d["successes"].size:
        out["solved"] = float(d["successes"][-1].mean())
        out["best_solved"] = float(d["successes"].mean(1).max())
    return out


def make_video(run, step_k, py):
    os.makedirs(os.path.join(run, "progress"), exist_ok=True)
    out = os.path.join(run, "progress", f"step_{step_k:05d}k.mp4")
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from scripts.evaluate import load, record\n"
            "record(load(%r, best=False), %r, %r, seed=7)\n") % (ROOT_REPO, run, run, out)
    subprocess.run([py, "-c", code], cwd=ROOT_REPO, timeout=900, capture_output=True,
                   env=dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl")))
    return out if os.path.exists(out) else None


def write_dashboard(root, runs):
    t = cpu_temp_c()
    rows = []
    for run in runs:
        name = os.path.basename(run)
        ev = last_eval(run) or {}
        curve = os.path.join(run, "curves.png")
        try:
            subprocess.run([sys.executable, os.path.join(ROOT_REPO, "scripts/plot_runs.py"), run, "--out", curve],
                           capture_output=True, timeout=120)
        except Exception:
            pass
        vids = sorted(glob.glob(os.path.join(run, "progress", "*.mp4")), reverse=True)
        rel = lambda p: os.path.relpath(p, root)  # noqa: E731
        num = ""
        if ev:
            num = (f"{ev['steps']:,} steps &middot; eval return {ev['return']:.1f} (best {ev['best_return']:.1f})"
                   + (f" &middot; solved {ev['solved']:.0%} (best {ev['best_solved']:.0%})" if "solved" in ev else ""))
        v = "".join(f'<figure><video src="{rel(p)}" controls muted preload="none"></video>'
                    f'<figcaption>{os.path.basename(p)[5:-4]}</figcaption></figure>' for p in vids[:12])
        img = f'<img src="{rel(curve)}?t={int(time.time())}">' if os.path.exists(curve) else ""
        rows.append(f"<section><h2>{html.escape(name)}</h2><p>{num or 'no evaluation yet'}</p>{img}"
                    f"<div class='v'>{v or '<p>first progress video after 100k steps</p>'}</div></section>")
    page = f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="120"><title>Fly-brain training</title>
<style>body{{font:15px system-ui;margin:16px;max-width:1100px;background:#fafafa;color:#222}}
img{{max-width:100%;border:1px solid #ddd}}.v{{display:flex;flex-wrap:wrap;gap:10px}}
figure{{margin:0;width:340px}}video{{width:100%}}figcaption{{font-size:13px;color:#555}}
section{{background:#fff;padding:12px;margin:12px 0;border:1px solid #e5e5e5;border-radius:8px}}</style>
<h1>Fly-brain training PC</h1>
<p>Updated {time.strftime('%Y-%m-%d %H:%M')} &middot; CPU {'%.0f &deg;C' % t if t else 'n/a'} &middot; refreshes every 2 min.
Progress videos: newest checkpoint, same test room each time, newest first.</p>
{''.join(rows) or '<p>No runs yet.</p>'}"""
    tmp = os.path.join(root, "index.html.tmp")
    open(tmp, "w").write(page)
    os.replace(tmp, os.path.join(root, "index.html"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/flybrain"))
    ap.add_argument("--every", type=int, default=100_000, help="steps between progress videos")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    state_f = os.path.join(args.root, "watch_state.json")
    while True:
        state = json.load(open(state_f)) if os.path.exists(state_f) else {}
        runs = sorted(r for r in glob.glob(os.path.join(args.root, "runs", "*"))
                      if os.path.exists(os.path.join(r, "config.json")))
        for run in runs:
            ev = last_eval(run)
            if not ev or not os.path.exists(os.path.join(run, "model.zip")):
                continue
            mark = ev["steps"] // args.every * args.every
            if mark >= args.every and mark > state.get(run, 0):
                if make_video(run, mark // 1000, sys.executable):
                    state[run] = mark
                    json.dump(state, open(state_f, "w"))
        write_dashboard(args.root, runs)
        if args.once:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
