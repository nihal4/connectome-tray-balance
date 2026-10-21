"""A high-quality presentation video of a trained brain (for LinkedIn, talks, the website).

Full-HD (1920 x 1080, or 1080 x 1080 / 1080 x 1350), anti-aliased, shadows, a slowly moving camera,
H.264 at high quality. A title card, the BEST brain solving several new layouts (found first with the
camera off, then replayed identically with it on, as in showcase.py), labels in the scene, the wrist
camera (full detail and what the brain receives) or the fly eye as insets, and a closing card with the
run's measured results. Nothing is staged: every clip is the brain acting on its own.

    python scripts/linkedin_video.py --run ~/flybrain/runs/<run> [--clips 3] [--tries 30] [--aspect 16:9]
Output: <run>/progress/linkedin_<steps>k.mp4 (or --out).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")

REPO = "github.com/nihal4/connectome-tray-balance"
TEXT = {    # env -> (title, one line on the task, label of the progress counter)
    "panda_pickripe": ("Picking only the ripe tomato",
                       "Three tomatoes on a truss: green, turning, ripe. Pick only the ripe one into the basket."),
    "panda_harvest": ("Picking a tomato off the plant",
                      "Grip the cherry tomato, snap its stalk, carry it to the basket."),
    "panda_smell": ("Ripe or rotten, by smell alone",
                    "Identical-looking fruit; the fly's smell receptors decide which bin."),
    "panda_sort": ("Sorting by colour", "Each block into the bin of its colour."),
    "panda_place": ("Pick and place", "Grasp the block, carry it into the bin."),
    "panda_lift": ("Grasp and lift", "Grasp the block and lift it."),
    "panda_reach": ("Reaching", "Bring the gripper to the yellow block."),
}


def font(size, bold=False):
    from PIL import ImageFont
    for f in (f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
              f"/usr/share/fonts/truetype/liberation/LiberationSans-{'Bold' if bold else 'Regular'}.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--clips", type=int, default=3, help="solved episodes to show")
    ap.add_argument("--tries", type=int, default=30, help="new layouts to try (camera off) to find them")
    ap.add_argument("--first_seed", type=int, default=60_000)
    ap.add_argument("--aspect", default="16:9", choices=["16:9", "1:1", "4:5"])
    ap.add_argument("--width", type=int, default=0, help="override the width (e.g. 960 for a quick test)")
    ap.add_argument("--author", default="S. M. Nihal Ahmed")
    ap.add_argument("--out", default=None)
    ap.add_argument("--any", action="store_true", help="(testing) use layouts even if not solved")
    args = ap.parse_args()

    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw
    from ctb.envs import build_env
    from scripts.evaluate import load, make_norm, run_config, run_env_kw

    run = os.path.abspath(os.path.expanduser(args.run))
    cfg = run_config(run)
    env_name = cfg["env"]
    model = load(run, best=True)
    vn = make_norm(run, {}, best=True)
    kw = run_env_kw(run)
    W, H = {"16:9": (1920, 1080), "1:1": (1080, 1080), "4:5": (1080, 1350)}[args.aspect]
    if args.width:
        H, W = int(round(H * args.width / W / 8)) * 8, args.width
    s = W / 1920 if args.aspect == "16:9" else W / 1080          # text scale
    fps = 25

    # ---- 1. find solved layouts (camera off; the brain is deterministic, so replays are identical) ----
    solved, n_tried = [], 0
    for i in range(args.tries):
        seed = args.first_seed + i
        env = build_env(env_name, seed, **kw)
        obs, _ = env.reset(seed=seed)
        info = {}
        for _ in range(env.max_steps):
            a, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
            obs, _, te, tr, info = env.step(a[0])
            if te or tr or info.get("is_success"):
                break
        n_tried += 1
        env.close()
        if info.get("is_success") or args.any:
            solved.append(seed)
            print(f"layout {seed}: solved", flush=True)
            if len(solved) >= args.clips:
                break
    if not solved:
        raise SystemExit("no solved layout found; try more --tries")

    # measured results for the closing card
    best = None
    ev = os.path.join(run, "evaluations.npz")
    if os.path.exists(ev):
        d = np.load(ev)
        if "successes" in d.files and len(d["successes"]):
            m = d["successes"].mean(1)
            best = (float(m.max()), d["successes"].shape[1])
    subset = np.load(os.path.join(run, "subset.npz"))
    n_neurons = int(subset["n"]) if "n" in subset.files else None
    n_syn = len(subset["pre"]) if "pre" in subset.files else None

    title, task_line = TEXT.get(env_name, (env_name, ""))
    steps_k = 0
    if os.path.exists(ev):
        steps_k = int(np.load(ev)["timesteps"][-1]) // 1000
    out = args.out or os.path.join(run, "progress", f"linkedin_{steps_k:05d}k{'' if args.aspect == '16:9' else '_' + args.aspect.replace(':', 'x')}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out[:-4] + ".part.mp4"
    writer = imageio.get_writer(tmp, fps=fps, codec="libx264", quality=None, macro_block_size=8,
                                pixelformat="yuv420p",
                                ffmpeg_params=["-crf", "17", "-preset", "slow", "-movflags", "+faststart"])

    def card(lines, frames, fade=10):
        """Dark title / closing card: lines = [(text, size, bold, colour)]."""
        img = Image.new("RGB", (W, H), (14, 17, 23))
        dr = ImageDraw.Draw(img)
        total = sum(int(sz * s * 1.55) for _, sz, _, _ in lines)
        y = (H - total) // 2
        for text, sz, bold, col in lines:
            f = font(int(sz * s), bold)
            w = dr.textlength(text, font=f)
            dr.text(((W - w) / 2, y), text, font=f, fill=col)
            y += int(sz * s * 1.55)
        base = np.asarray(img).astype(np.float32)
        for k in range(frames):
            a = min(1.0, (k + 1) / fade, (frames - k) / fade)
            writer.append_data((base * a).astype(np.uint8))

    def wrap(text, f, width, dr):
        words, lines, cur = text.split(), [], ""
        for w_ in words:
            t = (cur + " " + w_).strip()
            if dr.textlength(t, font=f) > width and cur:
                lines.append(cur)
                cur = w_
            else:
                cur = t
        return lines + ([cur] if cur else [])

    brain = f"{n_neurons:,} neurons, {n_syn:,} synapses" if n_neurons and n_syn else "fly brain wiring"
    card([("A fruit fly's brain wiring", 54, True, (240, 242, 245)),
          ("controlling a robot arm", 54, True, (240, 242, 245)),
          ("", 20, False, (0, 0, 0)),
          (title, 40, False, (120, 200, 140)),
          ("", 16, False, (0, 0, 0)),
          (f"FlyWire connectome  ·  {brain}  ·  MuJoCo  ·  Franka Panda", 26, False, (170, 176, 186)),
          (args.author, 26, False, (170, 176, 186))], frames=int(3.5 * fps))

    # ---- 2. replay the solved layouts with a good camera -----------------------------------------
    for ci, seed in enumerate(solved):
        env = build_env(env_name, seed, **kw)
        m = env.model
        m.vis.global_.offwidth, m.vis.global_.offheight = max(W, 640), max(H, 480)
        m.vis.quality.offsamples = 8
        m.vis.quality.shadowsize = 4096
        r = mujoco.Renderer(m, H, W)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = [1, 1, 1, 0, 1, 0]                  # no collision shapes, no eye "head" ball
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        harvest = getattr(env, "harvest", False)
        cam.lookat[:] = [0.52, -0.07, 0.22] if harvest else [0.48, 0.0, 0.10]
        cam.distance = (1.3 if harvest else 1.3) * (1.0 if args.aspect == "16:9" else 1.3)
        cam.elevation = -23
        az0 = (48 if harvest else 35) + (ci - 1) * 8        # each clip from a slightly different side
        insets = None
        if getattr(env, "wrist_cam", None):
            ins_w = int(400 * s)
            ins_h = int(ins_w * 0.625)
            insets = mujoco.Renderer(m, 200, 320)
        obs, _ = env.reset(seed=seed)
        info = {}
        frames_ep = []
        for t in range(env.max_steps):
            a, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
            obs, _, te, tr, info = env.step(a[0])
            cam.azimuth = az0 + 0.04 * t                      # a slow drift
            r.update_scene(env.data, camera=cam, scene_option=opt)
            frame = Image.fromarray(r.render())
            dr = ImageDraw.Draw(frame, "RGBA")
            el, az = np.radians(cam.elevation), np.radians(cam.azimuth)
            fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            cpos = np.array(cam.lookat) - cam.distance * fwd
            right = np.cross(fwd, [0, 0, 1.0])
            right /= np.linalg.norm(right)
            up = np.cross(right, fwd)
            foc = (H / 2) / np.tan(np.radians(m.vis.global_.fovy) / 2)

            def proj(p):
                v = np.asarray(p) - cpos
                zf = max(float(v @ fwd), 1e-6)
                return W / 2 + foc * float(v @ right) / zf, H / 2 - foc * float(v @ up) / zf

            f_lab, f_txt, f_small = font(int(24 * s), True), font(int(26 * s)), font(int(20 * s))
            if harvest:                                        # ripeness tags: a column left of the truss, with lines
                names = ("green", "turning", "ripe")
                cols = ((70, 170, 60), (235, 140, 40), (225, 45, 35))
                shown = [k for k in range(env.n_tom) if env.att[k]]
                pts = {k: proj(env.tom_pos(k)) for k in shown}
                if pts:
                    xl = min(p_[0] for p_ in pts.values()) - 190 * s
                    yc = float(np.mean([p_[1] for p_ in pts.values()]))
                    for i_, k in enumerate(sorted(shown, key=lambda k: pts[k][0])):
                        y = yc - 60 * s + i_ * 42 * s
                        txt = names[env.ripe[k]]
                        w_ = dr.textlength(txt, font=f_lab)
                        dr.line((xl + w_ + 16 * s, y + 15 * s, pts[k][0] - 10 * s, pts[k][1]), fill=cols[env.ripe[k]] + (230,),
                                width=max(2, int(3 * s)))
                        dr.rounded_rectangle((xl - 8 * s, y - 4 * s, xl + w_ + 8 * s, y + 32 * s),
                                             radius=int(8 * s), fill=(255, 255, 255, 225))
                        dr.text((xl, y), txt, font=f_lab, fill=cols[env.ripe[k]])
                bx, by = proj(env.bin_pos() + np.array([0, 0, 0.08]))
                w_ = dr.textlength("basket", font=f_lab)
                dr.text((bx - w_ / 2, by - 30 * s), "basket", font=f_lab, fill=(255, 255, 255),
                        stroke_width=max(1, int(3 * s)), stroke_fill=(90, 60, 30))
            # top-left: what is going on
            pad = int(24 * s)
            box_w = int(760 * s) if args.aspect == "16:9" else W - 2 * pad
            lines = wrap(task_line, f_txt, box_w - 2 * pad, dr)
            box_h = int(56 * s) + len(lines) * int(34 * s) + pad
            dr.rounded_rectangle((pad, pad, pad + box_w, pad + box_h), radius=int(14 * s), fill=(12, 14, 20, 200))
            dr.text((2 * pad, pad + int(14 * s)), title, font=font(int(32 * s), True), fill=(240, 242, 245))
            for li, ln in enumerate(lines):
                dr.text((2 * pad, pad + int(58 * s) + li * int(34 * s)), ln, font=f_txt, fill=(200, 205, 212))
            # bottom-left: progress
            if harvest and env.n_tom > 1:
                ripe = [k for k in range(env.n_tom) if env.ripe[k] == 2]
                status = f"ripe tomatoes in the basket: {sum(env.tom_in_basket(k) for k in ripe)} / {len(ripe)}"
            elif harvest:
                status = "in the basket" if env.in_bin() else "on the plant" if env.attached else "picked"
            else:
                status = "solved" if info.get("is_success") else ""
            if status:
                w_ = dr.textlength(status, font=f_txt)
                dr.rounded_rectangle((pad, H - pad - int(52 * s), pad + w_ + 2 * pad, H - pad), radius=int(12 * s),
                                     fill=(12, 14, 20, 200))
                dr.text((2 * pad, H - pad - int(42 * s)), status, font=f_txt, fill=(240, 242, 245))
            w_ = dr.textlength(REPO, font=f_small)
            dr.text((W - pad - w_, H - pad - int(26 * s)), REPO, font=f_small, fill=(235, 238, 242),
                    stroke_width=max(1, int(2 * s)), stroke_fill=(20, 22, 28))
            # right: the robot's own view
            if insets is not None:
                insets.update_scene(env.data, camera="wrist_cam", scene_option=env._cam_opt)
                full = Image.fromarray(insets.render()).resize((ins_w, ins_h), Image.LANCZOS)
                small = np.clip(env._cam_rgb, 0, 1)
                small = Image.fromarray((small * 255).astype(np.uint8)).resize((ins_w, ins_h), Image.NEAREST)
                x0 = pad + int(6 * s)                          # left column, under the title box
                y0 = pad + box_h + pad + int(6 * s)
                for j, (im, cap) in enumerate(((full, "wrist depth camera"),
                                               (small, "what the fly brain receives: 16 x 10 px"))):
                    yy = y0 + j * (ins_h + int(48 * s))
                    dr.rounded_rectangle((x0 - 6 * s, yy - 6 * s, x0 + ins_w + 6 * s, yy + ins_h + int(40 * s)),
                                         radius=int(10 * s), fill=(12, 14, 20, 210))
                    frame.paste(im, (x0, int(yy)))
                    dr.text((x0 + 4 * s, yy + ins_h + 6 * s), cap, font=f_small, fill=(215, 220, 226))
            frames_ep.append(np.asarray(frame))
            if info.get("is_success") or te or tr:
                break
        hold = frames_ep[-1]
        frames_ep += [hold] * int(1.2 * fps)
        for k, fr in enumerate(frames_ep):                    # fade in / out
            a = min(1.0, (k + 1) / 8, (len(frames_ep) - k) / 8)
            writer.append_data((fr.astype(np.float32) * a).astype(np.uint8))
        r.close()
        if insets is not None:
            insets.close()
        env.close()
        print(f"clip {ci + 1}/{len(solved)}: layout {seed}, {len(frames_ep)} frames", flush=True)

    # ---- 3. closing card: measured results ------------------------------------------------------
    facts = []
    if best:
        facts.append(f"best test: {best[0]:.0%} of {best[1]} new layouts solved")
    facts.append(f"this video: the first {len(solved)} solved of {n_tried} new layouts tried")
    learn = "copied a scripted teacher (DAgger), then practised (PPO)" if cfg.get("imitation") else \
        "learned by trial and error (PPO)"
    lines = [("How it learned", 40, True, (240, 242, 245)), ("", 10, False, (0, 0, 0)),
             (learn, 28, False, (200, 205, 212))]
    lines += [(f, 28, False, (120, 200, 140)) for f in facts]
    lines += [("", 10, False, (0, 0, 0)),
              ("The wiring is the fruit fly's own (FlyWire); connection strengths are learned.", 24, False, (170, 176, 186)),
              (REPO, 26, True, (240, 242, 245))]
    card(lines, frames=int(5 * fps))
    writer.close()
    os.replace(tmp, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
