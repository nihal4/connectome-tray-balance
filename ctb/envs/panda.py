"""Fly-brain Panda: a Franka Emika Panda arm with its two-finger gripper, seen and
steered by the fly-brain network.

Design follows the standard manipulation benchmarks (Fetch, robosuite, ManiSkill):
    * the brain commands the HAND, not the joints: [dx, dy, dz, grip] in [-1, 1],
      25 times a second. dx/dy/dz move the gripper up to 2 cm per step (0.5 m/s);
      grip > 0 opens, < 0 closes. A built-in controller (inverse kinematics + the
      arm's position servos) turns that into joint motions, like the fly's nerve cord
      turning descending commands into leg movements.
    * short episodes (60 steps = 2.4 s for reach), bounded staged rewards in the
      ManiSkill style, success = within 3 cm and still.

The brain sees the table through the same compound eye as the fly-brain drone
(two eyes x 160 facets, R1-6 / R7 / R8 channels), mounted on a "head" above and
behind the arm, looking down at the workspace. It feels the gripper position,
velocity and opening (ascending neurons) and never gets the block position.

Tasks (task=...):
    reach   bring the gripper to the yellow block (new place every episode) and stay
            just above it, ready to grasp (the pre-grasp pose: fingertip centre 1.5 cm
            above the block's top), within 3 cm for 0.2 s. Up to 2 blue decoy blocks.

Needs the Panda model: python scripts/download_panda.py
"""
from __future__ import annotations

import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .drone import N_CH, _blur_rays, eye_facets

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PANDA_XML = os.environ.get("PANDA_XML", os.path.join(ROOT, "data", "menagerie", "franka_emika_panda", "panda.xml"))

STEP = 0.02                  # max hand move per brain step (m) -> 0.5 m/s at 25 Hz
FRAME_SKIP = 20              # physics 500 Hz, brain 25 Hz
WS_LO = np.array([0.30, -0.30, 0.015])     # where the gripper (TCP) may go
WS_HI = np.array([0.70, 0.30, 0.45])
BLOCK_LO = np.array([0.35, -0.22])         # where blocks appear
BLOCK_HI = np.array([0.65, 0.22])
BLOCK_H = 0.025              # half size of the 5 cm blocks (the gripper opens to 8 cm)
# Two eyes, like a fixed overview camera + a wrist camera in robot labs:
#   right eye: on a "head" beside the table, its field centred on the workspace (find the block)
#   left eye:  on the wrist, looking past the fingers (line the gripper up precisely)
HEAD_POS = np.array([0.50, -0.55, 0.45])
HEAD_LOOK_AT = np.array([0.50, 0.0, 0.03])
WRIST_EYE_POS = np.array([0.06, 0.0, 0.07])    # in the hand frame, beside the fingers
HOME = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])

# task (reach): success = gripper within 3 cm of the target for 5 steps in a row (0.2 s).
# A dwell, as in standard reaching tasks. An earlier version also required speed < 5 cm/s,
# but PPO's exploration noise alone moves the hand ~12 cm/s, so even a perfect pilot with
# that noise succeeded only 27 % of the time and training almost never saw a success.
SOLVE_DIST = 0.03
HOLD_STEPS = 5

# what each photoreceptor type sees: (R1-6 brightness, R7 UV, R8 green)
TABLE = np.array([0.35, 0.30, 0.35])     # grey
ROBOT = np.array([0.80, 0.70, 0.80])     # white-grey robot
WALLS = np.array([0.55, 0.60, 0.50])     # background (no hit): pale, a bit UV
TARGET = np.array([1.0, 0.05, 1.0])      # yellow block: green, almost no UV (unique)
DECOY = np.array([0.75, 0.90, 0.15])     # blue-violet decoy block


def _turn(u, v):
    """Quaternion of the smallest rotation that turns direction u into direction v."""
    ax = np.cross(u, v)
    s = np.linalg.norm(ax)
    ang = np.arctan2(s, float(u @ v))
    ax = ax / s if s > 1e-9 else np.array([1.0, 0, 0])
    return np.array([np.cos(ang / 2), *(np.sin(ang / 2) * ax)])


def _build_model(n_decoys, n_facets):
    spec = mujoco.MjSpec.from_file(PANDA_XML)
    wb = spec.worldbody
    light = wb.add_light()
    light.pos, light.dir = [0.3, 0, 2.0], [0, 0, -1]
    tex = spec.add_texture()
    tex.name, tex.type, tex.builtin = "tbl", mujoco.mjtTexture.mjTEXTURE_2D, mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex.rgb1, tex.rgb2, tex.width, tex.height = [0.55, 0.55, 0.55], [0.45, 0.45, 0.45], 64, 64
    mat = spec.add_material()
    mat.name = "tbl"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "tbl"
    mat.texrepeat = [10, 10]
    g = wb.add_geom()
    g.name, g.type, g.size, g.material = "table", mujoco.mjtGeom.mjGEOM_PLANE, [1.2, 1.2, 0.05], "tbl"

    def block(name, rgba):
        b = wb.add_body()
        b.name, b.pos = name, [0.5, 0, BLOCK_H]
        b.add_freejoint()
        bg = b.add_geom()
        bg.name, bg.type, bg.size, bg.rgba = name, mujoco.mjtGeom.mjGEOM_BOX, [BLOCK_H] * 3, rgba
        bg.mass, bg.group, bg.friction = 0.05, 4, [1.0, 0.005, 0.0001]
        return b

    block("block", [0.95, 0.85, 0.1, 1])
    for i in range(n_decoys):
        block(f"decoy{i}", [0.3, 0.3, 0.95, 1])
    right, _, _, _ = eye_facets(n_facets)
    c_right = right.mean(0) / np.linalg.norm(right.mean(0))           # centre of the right eye's field
    c_left = c_right * np.array([1, -1, 1])
    head = wb.add_body()
    head.name, head.pos = "head", HEAD_POS
    look = HEAD_LOOK_AT - HEAD_POS
    head.quat = _turn(c_right, look / np.linalg.norm(look))
    ws = spec.body("hand").add_site()
    ws.name, ws.pos = "wrist_eye", WRIST_EYE_POS
    ws.quat = _turn(c_left, np.array([0, 0, 1.0]))                     # field centred along the fingers
    hg = head.add_geom()
    hg.type, hg.size, hg.rgba = mujoco.mjtGeom.mjGEOM_SPHERE, [0.04, 0, 0], [0.6, 0.15, 0.1, 1]
    hg.contype = hg.conaffinity = 0
    hg.group = 5
    sky = spec.add_texture()
    sky.name, sky.type, sky.builtin = "sky", mujoco.mjtTexture.mjTEXTURE_SKYBOX, mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1, sky.rgb2, sky.width, sky.height = [0.75, 0.82, 0.9], [0.35, 0.4, 0.5], 64, 64
    centre = wb.add_body()
    centre.name, centre.pos = "ws_centre", [0.45, 0.0, 0.18]
    cam = wb.add_camera()
    cam.name, cam.pos = "overview", [1.55, 0.95, 1.05]
    cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    cam.targetbody = "ws_centre"
    cam.fovy = 48
    site = spec.body("hand").add_site()
    site.name, site.pos = "tcp", [0, 0, 0.1034]
    return spec.compile()


class FlyPandaEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, task="reach", n_facets=160, max_steps=60, seed=None, render_mode=None,
                 eye_noise=0.01, body_noise=0.005, light_jitter=0.2, max_distractors=2,
                 target_visible=True, random_start=0.0, cmd_smooth=1.0, edge_bias=0.0, **_):
        assert task in ("reach",), task
        if not os.path.exists(PANDA_XML):
            raise FileNotFoundError(f"{PANDA_XML} not found. Run: python scripts/download_panda.py")
        self.task, self.max_steps, self.render_mode = task, max_steps, render_mode
        self.eye_noise, self.body_noise, self.light_jitter = eye_noise, body_noise, light_jitter
        self.max_dist, self.target_visible = max_distractors, target_visible
        self.n_decoys = 2
        self.model = _build_model(self.n_decoys, n_facets)
        self.data = mujoco.MjData(self.model)
        self._ik = mujoco.MjData(self.model)
        m = self.model
        self.frame_skip = FRAME_SKIP
        self.tcp = m.site("tcp").id
        self.head = m.body("head").id
        self.hand = m.body("hand").id
        self.wrist_eye = m.site("wrist_eye").id
        self.block_q = m.jnt_qposadr[m.body("block").jntadr[0]]
        self.decoy_q = [m.jnt_qposadr[m.body(f"decoy{i}").jntadr[0]] for i in range(self.n_decoys)]
        self.block_geom = m.geom("block").id
        self.decoy_geoms = [m.geom(f"decoy{i}").id for i in range(self.n_decoys)]
        self.arm_dof = np.arange(7)
        self.jnt_lo, self.jnt_hi = m.jnt_range[:7, 0], m.jnt_range[:7, 1]
        self.finger_q = [m.jnt_qposadr[m.joint("finger_joint1").id], m.jnt_qposadr[m.joint("finger_joint2").id]]

        # compound eye (same as the drone): rays see the table and the robot (collision
        # shapes, group 3); blocks are blended in through each facet's blur profile
        self.n_facets = n_facets
        right, az, el, self.spacing = eye_facets(n_facets)
        left = right * np.array([1, -1, 1])
        self.facet_dirs = np.concatenate([right, left])
        self.facet_az, self.facet_el = np.concatenate([az, -az]), np.concatenate([el, el])
        self.rays, self.ray_w = _blur_rays(self.facet_dirs, self.spacing)
        self.n_rays = len(self.rays)
        nf = 2 * n_facets
        k = np.arange(self.n_rays)
        # rays of the right eye (facets 0..F-1) and of the left eye, in facet-major blocks
        self._eye_rays = [k[(k % nf) < n_facets], k[(k % nf) >= n_facets]]
        self._gid = np.zeros(self.n_rays, np.int32)
        self._dist = np.zeros(self.n_rays)
        self._ray_groups = np.array([1, 1, 0, 1, 0, 0], np.uint8)
        self._occl_groups = np.array([1, 1, 0, 1, 1, 0], np.uint8)
        self.sigma = self.spacing / 2.355
        self.spectrum = np.tile(ROBOT, (m.ngeom, 1))
        self.spectrum[m.geom("table").id] = TABLE
        self.table_id = m.geom("table").id

        self.n_vis = 2 * n_facets * N_CH
        self.observation_space = spaces.Box(-np.inf, np.inf, (2 * self.n_vis + 13,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.random_start = random_start
        self.edge_bias = edge_bias          # training only: share of blocks put near the table edges / corners
        # hand commands are averaged over the last few steps (like a real arm controller):
        # 1.0 = no smoothing, 0.3 = each step moves 30 % towards the new command
        self.cmd_smooth = cmd_smooth
        self.rng = np.random.default_rng(seed)
        self._renderer = None
        self._light, self._tile = 1.0, 0.1

    # ------------------------------------------------------------------ eye
    def _eyes(self):
        """(origin, rotation, body to ignore) of the right eye (head) and left eye (wrist)."""
        d = self.data
        return [(d.xpos[self.head].copy(), d.xmat[self.head].reshape(3, 3), self.head),
                (d.site_xpos[self.wrist_eye].copy(), d.site_xmat[self.wrist_eye].reshape(3, 3), self.hand)]

    def eye_image(self):
        """(2 * n_facets, 3): right eye (head) then left eye (wrist); per facet the light
        seen by R1-6 (brightness), R7 (UV) and R8 (green)."""
        m, d = self.model, self.data
        F = self.n_facets
        n_f = 2 * F
        img = np.zeros((n_f, N_CH))
        world_center = np.zeros((n_f, 3))
        for e, (origin, R, skip) in enumerate(self._eyes()):
            idx = self._eye_rays[e]
            world = self.rays[idx] @ R.T
            n = len(idx)
            gid, dist = self._gid[:n], self._dist[:n]
            mujoco.mj_multiRay(m, d, origin, world.ravel(), self._ray_groups, 1, skip, gid, dist, None, n, 3.0)
            val = np.empty((n, N_CH))
            hit = gid >= 0
            val[~hit] = WALLS
            val[hit] = self.spectrum[gid[hit]]
            tb = gid == self.table_id
            if tb.any():                             # checker on the table (own-motion cues)
                p = origin + world[tb] * dist[tb, None]
                c = (np.floor(p[:, 0] / self._tile) + np.floor(p[:, 1] / self._tile)) % 2
                val[tb] *= (0.9 + 0.2 * c)[:, None]
            img[e * F:(e + 1) * F] = (val.reshape(-1, F, N_CH) * self.ray_w[:, None, None]).sum(0)
            world_center[e * F:(e + 1) * F] = world[:F]
            # blocks: blended into every facet by how much of its blurry view they fill
            objs = [(self.block_geom, TARGET)] if self.target_visible else []
            objs += [(g, DECOY) for g in self.decoy_geoms if d.geom_xpos[g][2] > -1]
            r_eq = BLOCK_H * 1.2                      # a cube seen as a ball of about the same size
            gid1 = np.zeros(1, np.int32)
            sl = slice(e * F, (e + 1) * F)
            fdir = world[:F]
            for g, spec in sorted(objs, key=lambda o: -np.linalg.norm(d.geom_xpos[o[0]] - origin)):
                rel = d.geom_xpos[g] - origin
                dist_b = np.linalg.norm(rel)
                hit_d = mujoco.mj_ray(m, d, origin, rel / dist_b, self._occl_groups, 1, skip, gid1)
                if gid1[0] != g and 0 <= hit_d < dist_b - 0.03:      # hidden behind the arm
                    continue
                alpha = np.arcsin(min(r_eq / dist_b, 0.99))
                theta = np.arccos(np.clip(fdir @ (rel / dist_b), -1, 1))
                s2 = self.sigma ** 2
                amp = 1 - np.exp(-alpha ** 2 / (2 * s2))
                wgt = amp * np.exp(-theta ** 2 / (2 * (s2 + alpha ** 2 / 4)))
                img[sl] = img[sl] * (1 - wgt[:, None]) + wgt[:, None] * spec
        self._world_dirs = world_center
        return np.clip(img * self._light, 0, 1.5)

    # ------------------------------------------------------------------ helpers
    def tcp_pos(self):
        return self.data.site_xpos[self.tcp].copy()

    def tcp_vel(self):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.tcp, v, 0)
        return v[3:].copy()

    def block_pos(self):
        return self.data.qpos[self.block_q:self.block_q + 3].copy()

    def reach_point(self):
        """Pre-grasp point: just above the block, where a grasp would start."""
        return self.block_pos() + np.array([0, 0, BLOCK_H + 0.015])

    def gripper_width(self):
        return float(self.data.qpos[self.finger_q[0]] + self.data.qpos[self.finger_q[1]])

    def _ik_solve(self, target, q0, iters=12):
        """Joint angles that put the TCP at target, gripper pointing straight down
        (damped least squares, pulled gently towards the home posture)."""
        m, d = self.model, self._ik
        d.qpos[:] = self.data.qpos
        q = q0.copy()
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        R_des = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])   # z down, fixed yaw
        for _ in range(iters):
            d.qpos[:7] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            mujoco.mj_jacSite(m, d, jacp, jacr, self.tcp)
            R = d.site_xmat[self.tcp].reshape(3, 3)
            e_pos = target - d.site_xpos[self.tcp]
            E = R_des @ R.T
            e_rot = 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
            J = np.vstack([jacp[:, :7], jacr[:, :7]])
            e = np.concatenate([e_pos, 0.5 * e_rot])
            JJt = J @ J.T + 1e-4 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            null = np.eye(7) - np.linalg.pinv(J) @ J
            dq += null @ (0.05 * (HOME - q))
            q = np.clip(q + dq, self.jnt_lo, self.jnt_hi)
            if np.linalg.norm(e_pos) < 1e-4:
                break
        return q

    def _get_obs(self):
        img = self.eye_image()
        if self.eye_noise > 0:
            img = img + self.rng.normal(0, self.eye_noise, img.shape)
        img = img.ravel()
        d_img = img - (self._prev_img if self._prev_img is not None else img)
        self._prev_img = img
        p, v = self.tcp_pos(), self.tcp_vel()
        if self.body_noise > 0:
            p = p + self.rng.normal(0, self.body_noise, 3)
            v = v + self.rng.normal(0, 2 * self.body_noise, 3)
        body = np.concatenate([(p - 0.5 * (WS_LO + WS_HI)) * 4, v * 2, [self.gripper_width() * 25 - 1],
                               [float(self._grip_ctrl > 127)], [0.0], self._last_action])
        return np.concatenate([img, d_img, body]).astype(np.float32)

    # ------------------------------------------------------------------ gym API
    def _place(self, qadr, xy):
        self.data.qpos[qadr:qadr + 3] = [xy[0], xy[1], BLOCK_H]
        yaw = self.rng.uniform(-np.pi, np.pi)
        self.data.qpos[qadr + 3:qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, 0)
        self._light = 1 + self.rng.uniform(-self.light_jitter, self.light_jitter)
        self._tile = self.rng.uniform(0.06, 0.14)
        target = self.rng.uniform(BLOCK_LO, BLOCK_HI)
        if self.edge_bias > 0 and self.rng.random() < self.edge_bias:
            # practise the weak spots: a 7 cm band along one edge, and half the time a corner
            band = 0.07
            axes = [self.rng.integers(2)]
            if self.rng.random() < 0.5:
                axes = [0, 1]
            for ax in axes:
                target[ax] = (BLOCK_HI[ax] - self.rng.uniform(0, band)) if self.rng.random() < 0.5 \
                    else (BLOCK_LO[ax] + self.rng.uniform(0, band))
        self._place(self.block_q, target)
        n_d = self.rng.integers(0, self.max_dist + 1) if self.max_dist > 0 else 0
        placed = [target]
        for i, qa in enumerate(self.decoy_q):
            if i < n_d:
                for _ in range(100):
                    xy = self.rng.uniform(BLOCK_LO, BLOCK_HI)
                    if min(np.linalg.norm(xy - p) for p in placed) > 0.1:
                        break
                placed.append(xy)
                self._place(qa, xy)
            else:
                d.qpos[qa:qa + 3] = [0, 0, -5]              # parked under the table, unseen
        # gripper starts somewhere above the table (or near the block: curriculum)
        start = self.rng.uniform([0.35, -0.2, 0.18], [0.65, 0.2, 0.35])
        if self.random_start > 0 and self.rng.random() < self.random_start:
            start = np.array([*target, 0.0]) + self.rng.uniform([-0.08, -0.08, 0.05], [0.08, 0.08, 0.12])
        q = self._ik_solve(start, HOME, iters=60)
        d.qpos[:7] = q
        d.ctrl[:7] = q
        d.ctrl[7] = self._grip_ctrl = 255.0
        mujoco.mj_forward(m, d)
        for _ in range(50):                                 # let blocks settle
            mujoco.mj_step(m, d)
        if n_d < len(self.decoy_q):                          # keep parked decoys parked
            for qa in self.decoy_q[n_d:]:
                d.qpos[qa:qa + 3] = [0, 0, -5]
                d.qvel[:] = 0
            mujoco.mj_forward(m, d)
        self._target = self.tcp_pos()
        self._cmd = np.zeros(3)
        self._hold = 0
        self._last_action = np.zeros(4)
        self._steps = 0
        self._prev_img = None
        return self._get_obs(), {}

    def _reward(self):
        d_t = float(np.linalg.norm(self.tcp_pos() - self.reach_point()))
        inside = d_t < SOLVE_DIST
        self._hold = self._hold + 1 if inside else 0
        r = 1 - np.tanh(5 * d_t)                              # reach
        if inside:
            r += 1.0                                          # in the target zone
        success = self._hold >= HOLD_STEPS                    # ... and stayed there 0.2 s
        if success:
            r = 3.0
        return r / 3.0, success, d_t

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        m, d = self.model, self.data
        # move a target point that the hand follows (like Fetch's mocap target), kept
        # within 5 cm of the real hand so it cannot run away when the arm is blocked
        tcp = self.tcp_pos()
        self._cmd = self._cmd + self.cmd_smooth * (a[:3] - self._cmd)
        self._target = np.clip(self._target + self._cmd * STEP, WS_LO, WS_HI)
        off = self._target - tcp
        n = np.linalg.norm(off)
        if n > 0.05:
            self._target = tcp + off * 0.05 / n
        d.ctrl[:7] = self._ik_solve(self._target, d.qpos[:7].copy())
        self._grip_ctrl = float(np.clip(self._grip_ctrl + a[3] * 64, 0, 255))   # 4 steps to open/close
        d.ctrl[7] = self._grip_ctrl
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(m, d)
        self._steps += 1
        self._last_action = a
        r, success, d_t = self._reward()
        truncated = self._steps >= self.max_steps
        info = {"is_success": success, "dist": d_t}
        return self._get_obs(), float(r), False, truncated, info

    # ------------------------------------------------------------------ rendering
    def eye_view(self, size=190):
        img = self.eye_image()
        F = self.n_facets
        h = size // 2
        out = np.zeros((h + 4, 2 * size + 6, 3))
        for e, x0 in ((1, 0), (0, size + 6)):
            sl = slice(e * F, (e + 1) * F)
            az, el, val = self.facet_az[sl], self.facet_el[sl], img[sl]
            u = (az - az.min()) / (np.ptp(az) + 1e-9)
            if e == 0:
                u = 1 - u
            xs = (x0 + u * (size - 5)).astype(int) + 2
            ys = ((1 - (el - el.min()) / (np.ptp(el) + 1e-9)) * (h - 3)).astype(int) + 2
            rgb = np.clip(np.stack([val[:, 0], val[:, 2], val[:, 1]], 1), 0, 1)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    out[np.clip(ys + dy, 0, h + 3), np.clip(xs + dx, 0, 2 * size + 5)] = rgb
        return (out * 255).astype(np.uint8)

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
            self._vopt = mujoco.MjvOption()
            self._vopt.geomgroup[:] = [1, 1, 1, 0, 1, 1]
        self._renderer.update_scene(self.data, camera="overview", scene_option=self._vopt)
        frame = self._renderer.render().copy()
        eye = self.eye_view()
        frame[6:6 + eye.shape[0], 6:6 + eye.shape[1]] = eye
        return frame

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def panda_controller(env, obs=None):
    """Hand-written solution using the TRUE block position (the brain never gets it)."""
    err = env.reach_point() - env.tcp_pos()
    a = np.zeros(4)
    a[:3] = np.clip(err / STEP * 0.5, -1, 1)
    a[3] = 1.0
    return a
