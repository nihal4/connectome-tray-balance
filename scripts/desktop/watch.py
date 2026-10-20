"""Live dashboard + progress videos for the training PC (runs as the flybrain-watch service).

Every 10 s: read only the NEW lines of the training log and write ~/flybrain/status.json
(a few KB: live numbers, learning-curve data, video list, CPU temperature, RAM). The page
~/flybrain/index.html (copied from dashboard.html) fetches it and redraws in the browser, so
the PC does no plotting for the dashboard. The flybrain-web service serves it on port 8080.

Every 60 s: when a run has passed another 100k steps, record the newest checkpoint on the same
3 test layouts and save runs/<run>/progress/step_<N>k.mp4; redraw runs/<run>/curves.png (used
by `fly export`) only when a new evaluation has arrived.

Runs at the lowest priority with 1 thread, so it barely slows training.
"""
import argparse
import glob
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


def start_video(run, step_k, py):
    """Record a progress video in the background (the dashboard keeps updating meanwhile).
    Returns (process, output path, start time)."""
    os.makedirs(os.path.join(run, "progress"), exist_ok=True)
    out = os.path.join(run, "progress", f"step_{step_k:05d}k.mp4")
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from scripts.desktop.watch import record_tests\n"
            "record_tests(%r, %r)\n") % (ROOT_REPO, run, out)
    proc = subprocess.Popen([py, "-c", code], cwd=ROOT_REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env=dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl")),
                            preexec_fn=lambda: os.nice(10))
    return proc, out, time.time()


TEST_SEEDS = (7, 8, 9)      # the same three test rooms / mazes in every video, so videos compare
# Panda: three blocks spread across the table (left-near, centre, right-far), so the clips
# visibly show different positions: seed 1 = (0.39, +0.20), 24 = (0.52, 0.00), 4 = (0.64, -0.18)
TEST_SEEDS_BY_ENV = {"panda_reach": (1, 24, 4), "panda_lift": (1, 24, 4), "panda_place": (1, 24, 4), "panda_sort": (13, 11, 10), "panda_smell": (2, 5, 8), "panda_harvest": (1, 3, 7)}


def record_tests(run, out, seeds=None, best=False):
    """One video: a brain (newest checkpoint, or the best one) on test layouts, back to back,
    with a short dark gap between them. Episodes that are solved end early."""
    import imageio.v2 as imageio
    from ctb.envs import build_env
    from scripts.evaluate import add_eye_view, load, make_norm, run_config, run_env_kw
    model = load(run, best=best)
    env_name = run_config(run)["env"]
    vn = make_norm(run, {}, best=best)
    frames, results = [], []
    for seed in (seeds or TEST_SEEDS_BY_ENV.get(env_name, TEST_SEEDS)):
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


def read_job(root):
    """~/flybrain/job.env as a dict (ENV, STEPS, EXTRA, NAME, BRAIN)."""
    job = {}
    try:
        for line in open(os.path.join(root, "job.env")):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                job[k] = v.strip('"')
    except OSError:
        pass
    return job


def current_job(root):
    """Run name the training service works on."""
    return read_job(root).get("NAME")


def service_state(name="flybrain-train"):
    try:
        r = subprocess.run(["systemctl", "--user", "is-active", name], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def ram_free_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) / 1048576, 1)
    except OSError:
        pass
    return None


def _num(x):
    """Number from a log cell, or None (also for nan / inf, which would make the JSON unreadable)."""
    try:
        v = float(x.replace(",", "").replace("%", ""))
    except ValueError:
        return None
    return v if np.isfinite(v) else None


def _clean(o):
    """Replace nan / inf anywhere in the status by None, so the browser can always parse it."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


class LogTail:
    """Reads a training log incrementally (only bytes added since the last call) and keeps
    the latest numbers plus the training-return history from the [progress] lines."""

    KEYS = {"fps": "fps", "std": "std", "approx_kl": "approx_kl", "ep_rew_mean": "ep_rew_mean",
            "total_timesteps": "total_timesteps", "learning_rate": "learning_rate",
            "clip_fraction": "clip_fraction", "explained_variance": "explained_variance"}

    def __init__(self, path):
        self.path, self.pos, self.buf = path, 0, ""
        self.live, self.train = {}, {"t": [], "r": [], "s": []}
        self.events = []

    def update(self):
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return
        if size < self.pos:                       # log was replaced: start over
            self.__init__(self.path)
        if size == self.pos:
            return
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            chunk = f.read(size - self.pos).decode("utf-8", "replace")
        self.pos = size
        lines = (self.buf + chunk).split("\n")
        self.buf = lines.pop()                    # keep a half-written last line for next time
        for ln in lines:
            self._line(ln.strip())
        self.live["log_mtime"] = os.path.getmtime(self.path)

    def _line(self, ln):
        if ln.startswith("[progress]"):
            # [progress] 1,150,012 steps | training episodes: return 31.16 | solved 0%
            parts = ln.split("|")
            steps = _num(parts[0].split()[1])
            ret = sol = None
            for p in parts[1:]:
                if "return" in p:
                    ret = _num(p.split("return")[1].strip())
                if "solved" in p:
                    sol = _num(p.split("solved")[1].strip())
            if steps is not None and ret is not None:
                if self.train["t"] and steps <= self.train["t"][-1]:     # resumed: drop the overlap
                    while self.train["t"] and self.train["t"][-1] >= steps:
                        for k in self.train:
                            self.train[k].pop()
                self.train["t"].append(steps)
                self.train["r"].append(ret)
                self.train["s"].append(sol)
                self.live.update(train_steps=steps, train_return=ret, train_solved=sol)
        elif ln.startswith("|") and ln.count("|") >= 3:
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if len(cells) == 2 and cells[0] in self.KEYS:
                v = _num(cells[1])
                if v is not None:
                    self.live[self.KEYS[cells[0]]] = v
        elif ln.startswith("Eval num_timesteps="):
            try:
                self.live["eval_steps"] = int(ln.split("=")[1].split(",")[0])
                self.live["eval_return"] = float(ln.split("episode_reward=")[1].split()[0])
            except (IndexError, ValueError):
                pass
        elif ln.startswith("Success rate:"):
            self.live["eval_solved"] = _num(ln.split(":")[1].strip())
        elif ln.startswith(("[train]", "[safety]", "[imitate]")):
            self.events = (self.events + [ln[:200]])[-8:]


def eval_history(run):
    """Evaluation history of a run (from evaluations.npz) for the browser charts."""
    f = os.path.join(run, "evaluations.npz")
    try:
        d = np.load(f)
        t = d["timesteps"].tolist()
        r = np.nan_to_num(d["results"].mean(1), nan=0.0).round(2).tolist() if len(t) else []
        s = (d["successes"].mean(1) * 100).round(1).tolist() if "successes" in d.files and d["successes"].size else []
        return {"t": t, "r": r, "s": s, "n_episodes": int(d["results"].shape[1]) if len(t) else 0}
    except Exception:
        return {"t": [], "r": [], "s": [], "n_episodes": 0}


class Dashboard:
    def __init__(self, root):
        self.root = root
        self.tails = {}          # log path -> LogTail
        self.cache = {}          # run -> (evaluations.npz mtime, eval history)
        page = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
        dst = os.path.join(root, "index.html")
        if os.path.exists(page):
            tmp = dst + ".tmp"
            with open(page) as fi, open(tmp, "w") as fo:
                fo.write(fi.read())
            os.replace(tmp, dst)

    def run_info(self, run, active):
        name = os.path.basename(run)
        f = os.path.join(run, "evaluations.npz")
        mt = os.path.getmtime(f) if os.path.exists(f) else 0
        if self.cache.get(run, (None,))[0] != mt:
            self.cache[run] = (mt, eval_history(run))
            if mt:                                 # new evaluation: refresh curves.png for `fly export`
                try:
                    subprocess.run([sys.executable, os.path.join(ROOT_REPO, "scripts/plot_runs.py"), run,
                                    "--out", os.path.join(run, "curves.png")], capture_output=True, timeout=120)
                except Exception:
                    pass
        ev = self.cache[run][1]
        vids = sorted(glob.glob(os.path.join(run, "progress", "*.mp4")), key=os.path.getmtime, reverse=True)
        log = os.path.join(self.root, "logs", name + ".log")
        train = {"t": [], "r": [], "s": []}
        if os.path.exists(log):
            if log not in self.tails:
                self.tails[log] = LogTail(log)
            self.tails[log].update()
            train = self.tails[log].train
        try:
            env = json.load(open(os.path.join(run, "config.json"))).get("env", "")
        except Exception:
            env = ""
        stamps = [mt] + [os.path.getmtime(v) for v in vids[:1]]
        if os.path.exists(log):
            stamps.append(os.path.getmtime(log))
        last = max(stamps)
        return {"name": name, "env": env, "active": name == active, "eval": ev, "train": train,
                "best_return": max(ev["r"]) if ev["r"] else None,
                "best_solved": max(ev["s"]) if ev["s"] else None,
                "videos": [os.path.relpath(v, self.root) for v in vids],
                "last_update": last}

    def write(self, runs):
        job = read_job(self.root)
        active = job.get("NAME")
        infos = [self.run_info(r, active) for r in runs]
        infos.sort(key=lambda i: (not i["active"], -i["last_update"]))
        live = {}
        events = []
        if active:
            tail = self.tails.get(os.path.join(self.root, "logs", active + ".log"))
            if tail:
                live, events = dict(tail.live), tail.events
        t = cpu_temp_c()
        status = {"updated": time.time(), "cpu_temp": round(t, 1) if t else None, "ram_free_gb": ram_free_gb(),
                  "service": service_state(), "job": {"name": active, "env": job.get("ENV"),
                  "steps_total": _num(job.get("STEPS", "") or "0"), "extra": job.get("EXTRA", "")},
                  "live": live, "events": events, "runs": infos}
        tmp = os.path.join(self.root, "status.json.tmp")
        with open(tmp, "w") as f:
            json.dump(_clean(status), f, separators=(",", ":"), allow_nan=False)
        os.replace(tmp, os.path.join(self.root, "status.json"))


def find_runs(root):
    return sorted(r for r in glob.glob(os.path.join(root, "runs", "*"))
                  if os.path.exists(os.path.join(r, "config.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/flybrain"))
    ap.add_argument("--every", type=int, default=100_000, help="steps between progress videos")
    ap.add_argument("--live", type=int, default=10, help="seconds between dashboard updates")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    state_f = os.path.join(args.root, "watch_state.json")
    dash = Dashboard(args.root)
    last_video_check = 0.0
    pending = None                          # (process, out, started, run, mark): one video at a time
    while True:
        runs = find_runs(args.root)
        if pending is not None:
            proc, out, t0, run, mark = pending
            if proc.poll() is None and time.time() - t0 > 1800:
                proc.kill()                 # stuck: give up on this one, try again at the next mark
            if proc.poll() is not None:
                state = json.load(open(state_f)) if os.path.exists(state_f) else {}
                state[run + "|tried"] = mark            # never retry the same mark over and over
                if os.path.exists(out):
                    state[run] = mark
                json.dump(state, open(state_f, "w"))
                pending = None
        if pending is None and time.time() - last_video_check >= 60:
            last_video_check = time.time()
            state = json.load(open(state_f)) if os.path.exists(state_f) else {}
            for run in runs:
                ev = last_eval(run)
                if not ev or not os.path.exists(os.path.join(run, "model.zip")):
                    continue
                if not os.path.exists(os.path.join(args.root, "logs", os.path.basename(run) + ".log")):
                    continue                    # a copied brain (e.g. panda_sort-v12), not a training run
                mark = ev["steps"] // args.every * args.every
                if mark >= args.every and mark > max(state.get(run, 0), state.get(run + "|tried", 0)):
                    pending = (*start_video(run, mark // 1000, sys.executable), run, mark)
                    break
        dash.write(runs)
        if args.once:
            break
        time.sleep(args.live)


if __name__ == "__main__":
    main()
