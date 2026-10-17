"""Progress videos + dashboard for the training PC (runs as the flybrain-watch service).

Every minute: for each run in ~/flybrain/runs, when training has passed another 100k steps,
run the newest checkpoint on 3 test mazes / rooms (always the same three, so the videos can
be compared, but different from each other) and save runs/<run>/progress/step_<N>k.mp4. Then rewrite ~/flybrain/index.html
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
            "from scripts.desktop.watch import record_tests\n"
            "record_tests(%r, %r)\n") % (ROOT_REPO, run, out)
    subprocess.run([py, "-c", code], cwd=ROOT_REPO, timeout=900, capture_output=True,
                   env=dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl")))
    return out if os.path.exists(out) else None


TEST_SEEDS = (7, 8, 9)      # the same three test rooms / mazes in every video, so videos compare
# Panda: three blocks spread across the table (left-near, centre, right-far), so the clips
# visibly show different positions: seed 1 = (0.39, +0.20), 24 = (0.52, 0.00), 4 = (0.64, -0.18)
TEST_SEEDS_BY_ENV = {"panda_reach": (1, 24, 4)}


def record_tests(run, out):
    """One video: the newest checkpoint on 3 different test mazes (rooms), back to back,
    with a short dark gap between them. Episodes that are solved end early."""
    import imageio.v2 as imageio
    from ctb.envs import build_env
    from scripts.evaluate import add_eye_view, load, make_norm, run_config, run_env_kw
    model = load(run, best=False)
    env_name = run_config(run)["env"]
    vn = make_norm(run, {})
    frames, results = [], []
    for seed in TEST_SEEDS_BY_ENV.get(env_name, TEST_SEEDS):
        env = build_env(env_name, seed, render_mode="rgb_array", **run_env_kw(run))
        obs, _ = env.reset(seed=seed)
        info = {}
        for _ in range(env.max_steps):
            a, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
            obs, _, te, tr, info = env.step(a[0])
            f = env.render()
            if getattr(env, "vision", False):
                f = add_eye_view(f, env.camera_frame())
            frames.append(f)
            if te or tr:
                break
        ok = bool(info.get("is_success", False))
        results.append(ok)
        mark = frames[-1].copy()                   # last frame: green border = solved, red = not
        mark[:10] = mark[-10:] = mark[:, :10] = mark[:, -10:] = (40, 180, 60) if ok else (200, 50, 50)
        frames += [mark] * 25 + [mark // 4] * 8
        env.close()
    fps = env.metadata.get("render_fps", 50)
    imageio.mimsave(out, frames, fps=fps)
    print(f"{out}: solved {sum(results)}/{len(results)}")


def current_job(root):
    """Run name from ~/flybrain/job.env (the run the training service works on)."""
    try:
        for line in open(os.path.join(root, "job.env")):
            if line.startswith("NAME="):
                return line.strip().split("=", 1)[1]
    except OSError:
        pass
    return None


def write_dashboard(root, runs):
    t = cpu_temp_c()
    active = current_job(root)
    # the run being trained first, then the others, most recently updated first
    def last_update(r):
        f = os.path.join(r, "progress.csv")
        return os.path.getmtime(f) if os.path.exists(f) else 0
    runs = sorted(runs, key=lambda r: (os.path.basename(r) != active, -last_update(r)))
    rows = []
    for i, run in enumerate(runs):
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
        badge = ("<span class='now'>training now</span>" if name == active
                 else "<span class='old'>paused / finished</span>")
        if i == 1 or (i == 0 and name != active):
            rows.append("<h2 class='earlier'>Earlier runs</h2>")
        rows.append(f"<section><h2>{html.escape(name)} {badge}</h2><p>{num or 'no evaluation yet (first one at 50k steps)'}</p>{img}"
                    f"<div class='v'>{v or '<p>first progress video after 100k steps</p>'}</div></section>")
    page = f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="120"><title>Fly-brain training</title>
<style>body{{font:15px system-ui;margin:16px;max-width:1100px;background:#fafafa;color:#222}}
img{{max-width:100%;border:1px solid #ddd}}.v{{display:flex;flex-wrap:wrap;gap:10px}}
figure{{margin:0;width:340px}}video{{width:100%}}figcaption{{font-size:13px;color:#555}}
section{{background:#fff;padding:12px;margin:12px 0;border:1px solid #e5e5e5;border-radius:8px}}
.now{{background:#1f8a3b;color:#fff;font-size:13px;padding:2px 8px;border-radius:10px;vertical-align:middle}}
.old{{background:#ddd;color:#444;font-size:13px;padding:2px 8px;border-radius:10px;vertical-align:middle}}
h2.earlier{{margin-top:28px;color:#666;font-size:18px}}</style>
<h1>Fly-brain training PC</h1>
<p>Updated {time.strftime('%Y-%m-%d %H:%M')} &middot; CPU {'%.0f &deg;C' % t if t else 'n/a'} &middot; refreshes every 2 min.
Progress videos: newest checkpoint on the same 3 test mazes/rooms each time (border at the end of each: green = solved, red = not), newest first.</p>
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
