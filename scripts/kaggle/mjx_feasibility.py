"""Can the phase 2c tomato scene run on a GPU with MuJoCo MJX?  (feasibility test, no training)

Measures, on whatever machine runs it (a Kaggle T4 via `fly kaggle mjx`, or locally):
  1. CPU baseline: how fast one CPU core steps our scene (what the training PC does, 4 cores).
  2. MJX speed: env steps per second for 64 ... 4096 arms stepped together on the GPU, for two versions
     of the scene:  "asis" (every collision shape as now) and "lite" (arm links do not collide, only the
     hand, fingers, tomatoes, plant, basket and table: how MJX robot scenes are usually set up).
  3. Cameras on the GPU: the two 16 x 10 depth/colour cameras done by ray casting (mjx.ray).
  4. Physics fidelity: the scripted teacher picks tomatoes on the CPU; the same motor commands (and stalk
     snaps) are replayed in MJX, and we compare where the tomatoes go and how many end in the basket.

Everything goes to OUT/report.json and OUT/summary.txt.  Nothing here needs any key or password.

    python scripts/kaggle/mjx_feasibility.py --out mjx_report                    # full test (GPU)
    python scripts/kaggle/mjx_feasibility.py --quick --out /tmp/mjx               # small local check (CPU)
"""
import argparse
import copy
import json
import os
import platform
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np                     # noqa: E402
import mujoco                          # noqa: E402
import jax                             # noqa: E402
import jax.numpy as jp                 # noqa: E402
from mujoco import mjx                 # noqa: E402

from ctb.envs import panda as P        # noqa: E402

# the phase 2c task as trained now (cameras are not built here: the CPU env is only the physics + teacher)
ENV_KW = dict(task="pickripe", many=True, gentle=True, grip_mode="absolute", privileged=True)
G = mujoco.mjtGeom
PC_TRAINING_FPS = 56           # the i3-8100 training PC on phase 2c (fly status), for comparison


def log(*a):
    print(*a, flush=True)


def make_env(seed):
    env = P.FlyPandaEnv(seed=seed, **ENV_KW)
    env.reset(seed=seed)
    return env


def mjx_model(m, variant):
    """A copy of the model that MJX can load. MJX has no cylinder-mesh or ellipsoid-mesh collisions, so the
    plant stem (cylinder) becomes a capsule and the solid leaves (flat ellipsoids) become flat boxes of the same
    size. The two unused decoy blocks of the older tasks (parked under the table, which the infinite floor plane shoots away: it blows up MJX) stop
    colliding and float. 'lite' also switches off the arm links' mesh collisions (hand, fingers and pads stay)."""
    m = copy.deepcopy(m)
    notes = {"stem_to_capsule": 0, "leaf_to_box": 0, "arm_links_off": 0, "decoys_off": 0, "grip_on_joint": 0}
    # the gripper: a tendon actuator + joint equality blows up in MJX; drive finger 1 directly (the same opening:
    # the tendon length is the mean of two equal fingers) and make the finger coupling a little softer
    if m.actuator_trntype[7] == mujoco.mjtTrn.mjTRN_TENDON:
        m.actuator_trntype[7] = mujoco.mjtTrn.mjTRN_JOINT
        m.actuator_trnid[7, 0] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        m.eq_solref[0] = np.maximum(m.eq_solref[0], [0.01, 1.0])
        notes["grip_on_joint"] = 1
    # the plant is fixed in space (mocap) and its stem reaches into the table: the CPU skips such fixed-fixed
    # pairs, MJX does not (a deep fake contact that blows up the solver). Collision bits: the plant only
    # meets the robot (as on the CPU: tomato-plant contacts are excluded there too)
    robot = ("link", "hand", "left_finger", "right_finger")
    for i in range(m.ngeom):
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]) or ""
        if body == "plant" and (m.geom_contype[i] or m.geom_conaffinity[i]):
            m.geom_contype[i] = m.geom_conaffinity[i] = 2
        elif body.startswith(robot) and m.geom_conaffinity[i]:
            m.geom_conaffinity[i] |= 2
    for b in range(m.nbody):
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("decoy"):
            m.body_gravcomp[b] = 1.0
    for i in range(m.ngeom):
        if not (m.geom_contype[i] or m.geom_conaffinity[i]):
            continue
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]) or ""
        if body.startswith("decoy"):
            m.geom_contype[i] = m.geom_conaffinity[i] = 0
            notes["decoys_off"] += 1
            continue
        if m.geom_type[i] == G.mjGEOM_CYLINDER:
            m.geom_type[i] = G.mjGEOM_CAPSULE
            notes["stem_to_capsule"] += 1
        elif m.geom_type[i] == G.mjGEOM_ELLIPSOID:
            m.geom_type[i] = G.mjGEOM_BOX
            notes["leaf_to_box"] += 1
        if variant == "lite" and m.geom_type[i] == G.mjGEOM_MESH and body.startswith("link"):
            m.geom_contype[i] = m.geom_conaffinity[i] = 0
            notes["arm_links_off"] += 1
    return m, notes


def state_for(m, src):
    """The simulation state of src (an MjData of the original model) as MjData of the changed model m."""
    d = mujoco.MjData(m)
    for f in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "eq_active"):
        getattr(d, f)[:] = getattr(src, f)
    d.time = src.time
    mujoco.mj_forward(m, d)
    return d


def system_info():
    dev = jax.devices()[0]
    return {"jax": jax.__version__, "mujoco": mujoco.__version__, "backend": jax.default_backend(),
            "device": getattr(dev, "device_kind", str(dev)), "n_devices": jax.device_count(),
            "cpu_count": os.cpu_count(), "cpu": platform.processor() or platform.machine(),
            "python": platform.python_version()}


# ---------------------------------------------------------------- 1. CPU baseline
def cpu_baseline(seconds=8.0):
    env = make_env(0)
    m, d = env.model, env.data
    n, t = 0, time.time()
    while time.time() - t < seconds:                 # physics only: 20 sub-steps = one 40 ms brain step
        for _ in range(P.FRAME_SKIP):
            mujoco.mj_step(m, d)
        n += 1
    phys = n / (time.time() - t)
    env = make_env(1)
    n, t = 0, time.time()
    while time.time() - t < seconds:                 # the whole env step (IK, reward, checks), no cameras
        _, _, term, trunc, _ = env.step(P.harvest_teacher(env))
        n += 1
        if term or trunc:
            env.reset()
    full = n / (time.time() - t)
    return {"physics_env_steps_per_s_1core": round(phys, 1), "env_steps_per_s_1core_no_cam": round(full, 1),
            "contacts_now": int(d.ncon)}


# ---------------------------------------------------------------- 2 + 3. MJX speed (+ ray cameras)
def cam_dirs(fovy_deg, w=16, h=10):
    """Unit view directions of a w x h camera in its own frame (MuJoCo: looks along -z, x right, y up)."""
    f = 0.5 * h / np.tan(np.radians(fovy_deg) / 2)
    u, v = np.meshgrid(np.arange(w) - w / 2 + 0.5, np.arange(h) - h / 2 + 0.5)
    dirs = np.stack([u.ravel(), -v.ravel(), -f * np.ones(w * h)], 1)
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def make_cameras(m, mx):
    """Both robot cameras as ray casts: per pixel (red, nearness, green), like the env's RGBD input."""
    cams = [(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, n), jp.asarray(cam_dirs(f)))
            for n, f in (("wrist_cam", P.CAM_FOVY), ("head_cam", P.HEADCAM_FOVY))]
    rgba = jp.asarray(m.geom_rgba)
    groups = (1, 0, 0, 0, 1, 0)        # what the eyes see: plant/bins/table (group 0) and tomatoes (group 4)

    def render(d):
        out = []
        for cid, dirs in cams:
            p, R = d.cam_xpos[cid], d.cam_xmat[cid].reshape(3, 3)
            dist, gid = jax.vmap(lambda v: mjx.ray(mx, d, p, R @ v, geomgroup=groups))(dirs)
            col = jp.where(gid[:, None] >= 0, rgba[gid, :3], 0.0)
            near = jp.where(gid >= 0, 1.0 - jp.clip(dist, 0.0, 1.0), 0.0)
            out.append(jp.stack([col[:, 0], near, col[:, 1]], 1))
        return jp.concatenate(out)                    # (320, 3)
    return render


def bench_variant(env, variant, batches, iters, impl, with_cams):
    res = {"variant": variant, "impl": impl}
    m, res["changes"] = mjx_model(env.model, variant)
    t = time.time()
    try:
        mx = mjx.put_model(m, impl=impl)
        dx = mjx.put_data(m, state_for(m, env.data), impl=impl)
    except Exception as e:                            # unsupported feature: that is the answer for this variant
        res["error"] = f"{type(e).__name__}: {e}"
        return res
    res["load_s"] = round(time.time() - t, 1)
    render = make_cameras(m, mx) if with_cams else None
    base = jp.asarray(env.data.ctrl)

    def one(d, ctrl):
        d = d.replace(ctrl=ctrl)
        d = jax.lax.fori_loop(0, P.FRAME_SKIP, lambda i, d: mjx.step(mx, d), d)
        return (d, render(d)) if render is not None else (d, jp.zeros(()))

    step = jax.jit(jax.vmap(one))
    res["runs"] = []
    for B in batches:
        r = {"batch": B}
        try:
            key = jax.random.PRNGKey(B)
            db = jax.tree_util.tree_map(lambda x: jp.broadcast_to(x, (B,) + x.shape) if hasattr(x, "shape") else x, dx)
            noise = 0.02 * jax.random.normal(key, (B, 7))
            db = db.replace(qpos=db.qpos.at[:, :7].add(noise))
            ctrls = [jp.broadcast_to(base, (B,) + base.shape).at[:, :7].add(
                0.03 * jax.random.normal(jax.random.fold_in(key, i), (B, 7))) for i in range(4)]
            t = time.time()
            db, img = step(db, ctrls[0])
            jax.block_until_ready(db.qpos)
            r["compile_s"] = round(time.time() - t, 1)
            t = time.time()
            for i in range(iters):
                db, img = step(db, ctrls[i % 4])
            jax.block_until_ready(db.qpos)
            dt = time.time() - t
            r["env_steps_per_s"] = round(B * iters / dt, 1)
            r["nan_envs"] = float(jp.mean(jp.any(~jp.isfinite(db.qpos), axis=1)))
            if render is not None:
                r["image_mean"] = [round(float(x), 3) for x in jp.mean(img, axis=(0, 1))]
            log(f"   {variant:5s} batch {B:5d}: {r['env_steps_per_s']:9.1f} env steps/s "
                f"(compile {r['compile_s']} s, broken envs {r['nan_envs']:.0%})")
        except Exception as e:                        # most likely out of GPU memory: stop growing the batch
            r["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            log(f"   {variant:5s} batch {B:5d}: failed ({r['error'][:120]})")
            res["runs"].append(r)
            break
        res["runs"].append(r)
    return res


# ---------------------------------------------------------------- 4. physics fidelity
def record_teacher(seed, max_steps=None):
    """One teacher episode on the CPU; records every physics sub-step's motor commands and stalk states."""
    env = make_env(seed)
    m0, d0 = copy.deepcopy(env.model), copy.copy(env.data)
    rec, orig = [], mujoco.mj_step

    def rec_step(m, d, *a):
        rec.append((d.ctrl.copy(), d.eq_active.copy()))
        return orig(m, d, *a)

    mujoco.mj_step = rec_step
    try:
        pos, info, n = [], {}, 0
        while True:
            _, _, term, trunc, info = env.step(P.harvest_teacher(env))
            pos.append([env.tom_pos(k) for k in range(env.n_tom)])
            n += 1
            if term or trunc or (max_steps and n >= max_steps):
                break
    finally:
        mujoco.mj_step = orig
    ctrl = np.array([c for c, _ in rec])
    eq = np.array([e for _, e in rec]).astype(bool)
    return env, m0, d0, ctrl, eq, np.array(pos), info


def replay(env, m0, d0, ctrl, eq, variant, impl):
    m, _ = mjx_model(m0, variant)
    mx = mjx.put_model(m, impl=impl)
    dx = mjx.put_data(m, state_for(m, d0), impl=impl)
    tq = jp.asarray(np.array([[q, q + 1, q + 2] for q in env.tom_q]))

    @jax.jit
    def run(dx, ctrl, eq):
        def sub(d, inp):
            c, e = inp
            d = mjx.step(mx, d.replace(ctrl=c, eq_active=e))
            return d, d.qpos[tq]
        return jax.lax.scan(sub, dx, (ctrl, eq))

    t = time.time()
    dfin, tpos = run(dx, jp.asarray(ctrl), jp.asarray(eq))
    jax.block_until_ready(dfin.qpos)
    return np.asarray(dfin.qpos), np.asarray(tpos)[P.FRAME_SKIP - 1::P.FRAME_SKIP], time.time() - t


def fidelity(seeds, variants, impl, max_steps=None):
    out = []
    for s in seeds:
        env, m0, d0, ctrl, eq, cpu_pos, info = record_teacher(s, max_steps)
        ripe = [k for k in range(env.n_tom) if env.ripe[k] == 2]
        present = [k for k in range(env.n_tom) if abs(float(cpu_pos[0][k][1])) < 1.0]   # parked ones are far away
        row = {"seed": s, "env_steps": len(cpu_pos), "cpu": {"ripe": len(ripe),
               "ripe_in_basket": int(info.get("ripe_in_basket", -1)), "solved": bool(info.get("is_success")),
               "bruised": int(info.get("bruised", 0))}}
        cpu_end = env.data.qpos.copy()
        for v in variants:
            try:
                q, mpos, sec = replay(env, m0, d0, ctrl, eq, v, impl)
                n = min(len(mpos), len(cpu_pos))
                err = np.linalg.norm(mpos[:n, present] - cpu_pos[:n, present], axis=-1)    # (steps, tomatoes)
                env.data.qpos[:] = q                     # score the MJX end state with the env's own basket check
                mujoco.mj_forward(env.model, env.data)
                in_b = int(sum(env.tom_in_basket(k) for k in ripe))
                env.data.qpos[:] = cpu_end
                mujoco.mj_forward(env.model, env.data)
                row[v] = {"ripe_in_basket": in_b, "finite": bool(np.isfinite(q).all()), "replay_s": round(sec, 1),
                          "tomato_err_cm_median": round(100 * float(np.median(err)), 2),
                          "tomato_err_cm_max": round(100 * float(np.max(err)), 2),
                          "tomato_err_cm_at_end": [round(100 * float(x), 2) for x in err[-1]],
                          "steps_until_err_over_2cm": int(np.argmax(err.max(1) > 0.02)) if (err.max(1) > 0.02).any() else None}
            except Exception as e:
                row[v] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            log(f"   seed {s} {v:5s}: CPU {row['cpu']['ripe_in_basket']}/{len(ripe)} in basket, "
                f"MJX {row[v].get('ripe_in_basket', '?')}/{len(ripe)}, "
                f"tomato error median {row[v].get('tomato_err_cm_median', '?')} cm, max {row[v].get('tomato_err_cm_max', '?')} cm")
        out.append(row)
    return out


def summary(rep):
    L = [f"MJX feasibility, {rep['system']['device']} ({rep['system']['backend']}), mujoco {rep['system']['mujoco']}, "
         f"jax {rep['system']['jax']}", ""]
    cb = rep.get("cpu_baseline", {})
    L.append(f"CPU, 1 core of this machine: physics {cb.get('physics_env_steps_per_s_1core')} env steps/s, "
             f"whole env without cameras {cb.get('env_steps_per_s_1core_no_cam')}")
    best = 0
    for v in rep.get("speed", []):
        if "error" in v:
            L.append(f"MJX {v['variant']}: cannot load: {v['error']}")
            continue
        for r in v["runs"]:
            if "env_steps_per_s" in r:
                best = max(best, r["env_steps_per_s"])
                L.append(f"MJX {v['variant']:5s}{' + cameras' if v.get('cams') else '':10s} batch {r['batch']:5d}: "
                         f"{r['env_steps_per_s']:9.1f} env steps/s, broken {r['nan_envs']:.0%}, compile {r['compile_s']} s")
            else:
                L.append(f"MJX {v['variant']:5s} batch {r['batch']:5d}: {r['error'][:100]}")
    for row in rep.get("fidelity", []):
        parts = [f"seed {row['seed']}: CPU {row['cpu']['ripe_in_basket']}/{row['cpu']['ripe']}"]
        for v in ("asis", "lite"):
            if v in row:
                x = row[v]
                parts.append(f"{v} {x.get('ripe_in_basket', '?')}/{row['cpu']['ripe']} "
                             f"(err median {x.get('tomato_err_cm_median', '?')} cm)" if "error" not in x else f"{v} error")
        L.append("fidelity " + ", ".join(parts))
    cams = [r["env_steps_per_s"] for v in rep.get("speed", []) if v.get("cams") for r in v.get("runs", [])
            if "env_steps_per_s" in r]
    if best:
        L += ["", f"training PC today: ~{PC_TRAINING_FPS} env steps/s (4 arms, cameras, fly brain and PPO)",
              f"MJX physics only, best: {best:.0f} env steps/s = {best / PC_TRAINING_FPS:.0f}x"]
        if cams:
            L.append(f"MJX physics + both cameras, best: {max(cams):.0f} env steps/s = {max(cams) / PC_TRAINING_FPS:.0f}x")
        L.append("(the brain and PPO also run on the GPU then, so training gains less; ~10x or more makes the rewrite "
                 "worth it, if the fidelity rows above match the CPU)")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mjx_report")
    ap.add_argument("--batches", default="64,256,1024,2048,4096")
    ap.add_argument("--variants", default="lite,asis")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--impl", default=None, help="MJX backend: jax (default) or warp")
    ap.add_argument("--quick", action="store_true", help="tiny sizes, to check the script on a CPU")
    a = ap.parse_args()
    if a.quick:
        a.batches, a.iters, a.seeds, a.variants = "2,4", 2, "0", "lite"
    os.makedirs(a.out, exist_ok=True)
    batches = [int(x) for x in a.batches.split(",")]
    variants = a.variants.split(",")
    rep = {"system": system_info(), "args": vars(a)}

    def save():
        with open(os.path.join(a.out, "report.json"), "w") as f:
            json.dump(rep, f, indent=1)
        with open(os.path.join(a.out, "summary.txt"), "w") as f:
            f.write(summary(rep) + "\n")

    log("system:", rep["system"])
    log("1. CPU baseline")
    rep["cpu_baseline"] = cpu_baseline(2.0 if a.quick else 8.0)
    log("  ", rep["cpu_baseline"])
    save()
    env = make_env(0)
    log("2. MJX speed, physics only")
    rep["speed"] = []
    for v in variants:
        t = time.time()
        rep["speed"].append(bench_variant(env, v, batches, a.iters, a.impl, with_cams=False))
        log(f"   ({v}: {time.time() - t:.0f} s)")
        save()
    log("3. MJX speed, physics + both ray cameras (lite)")
    try:
        r = bench_variant(env, "lite", batches, a.iters, a.impl, with_cams=True)
        r["cams"] = True
        rep["speed"].append(r)
    except Exception:
        rep["cams_error"] = traceback.format_exc()[-1500:]
        log(rep["cams_error"])
    save()
    log("4. physics fidelity: CPU teacher episode replayed in MJX")
    try:
        rep["fidelity"] = fidelity([int(s) for s in a.seeds.split(",")], variants, a.impl,
                                   max_steps=40 if a.quick else None)
    except Exception:
        rep["fidelity_error"] = traceback.format_exc()[-1500:]
        log(rep["fidelity_error"])
    save()
    log("\n" + summary(rep))


if __name__ == "__main__":
    main()
