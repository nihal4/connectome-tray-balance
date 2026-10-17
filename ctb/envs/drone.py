"""Fly-brain drone: a small quadcopter in a room, steered by the fly brain through a
fly-like compound eye.

Who does what (copied from the fly):
    halteres (balance organs)  ->  STABILISER here: keeps the drone level, holds the
                                   commanded velocity and heading (fast, 500 Hz)
    brain                      ->  the fly-brain network: decides where to fly (25 Hz)
    compound eye               ->  two eyes, a few hundred facets each, ray traced
    body sense                 ->  airspeed (antennae), tilt and rotation (halteres)

The brain sends 4 commands in [-1, 1]:
    forward, sideways (left +), up, turn (left +)
which the stabiliser turns into rotor thrusts. Speeds and accelerations are limited
(see the constants) so the drone moves smoothly, like a small real drone.

Task "reach": a bright target ball appears somewhere in the room (new place every
episode, start position and heading random too). Fly to it and hover on it for
0.4 s. The brain is NOT told where the target is: it has to find it with its eyes.

Reward (checked against shortcuts, see docs in README):
    +10 per metre closer to the target   (potential: going back and forth earns nothing)
    +0.5 per hold step on the target     (potential: leaving gives it back)
    +20 when solved (0.4 s on the target, slow)
    -0.01 per step (time)
    -10 and episode ends: crash (floor, wall, too high, flipped)
"""
from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# ---- flight limits (smooth, slow-ish flying; a real fly cruises at ~1 m/s)
V_XY = 1.0            # max horizontal speed, m/s
V_Z = 0.5             # max climb / sink speed, m/s
YAW_RATE = 1.5        # max turn rate, rad/s (~86 deg/s)
A_MAX = 2.0           # max change of commanded velocity, m/s^2 (no jerky starts/stops)
YAW_ACC = 6.0         # max change of commanded turn rate, rad/s^2
TILT_MAX = np.deg2rad(25)

# ---- drone and world
MASS = 0.5
ARM = 0.12            # rotor distance from centre (x and y), m
F_MAX = 2.5           # max thrust per rotor, N (thrust / weight = 2)
KM = 0.02             # rotor yaw torque per newton of thrust
ROOM = 4.0            # half size of the room, m (8 x 8 m)
WALL_H = 2.5
TARGET_R = 0.25
DT = 0.002            # physics step (500 Hz, the stabiliser runs every step)
FRAME_SKIP = 20       # brain step = 0.04 s (25 Hz)

# ---- task
SOLVE_DIST = 0.3      # drone centre within this distance of the target centre
SOLVE_SPEED = 0.5     # and slower than this (m/s)
HOLD_STEPS = 10       # for this many steps (0.4 s)
PROGRESS = 10.0
HOLD_POT = 0.5
SUCCESS = 20.0
CRASH = -10.0
TIME_COST = 0.01
JERK_COST = 0.0          # not needed: the stabiliser already limits acceleration,
                         # and a penalty here mostly punishes PPO's exploration noise

# spectral look of surfaces as (R1-6 brightness, R7 ultraviolet, R8 green)
SKY = np.array([0.85, 1.0, 0.55])        # sky is bright and UV-rich (flies use this)
FLOOR = np.array([0.30, 0.25, 0.30])     # grey concrete (UV and green alike)
WALL = np.array([0.45, 0.38, 0.45])      # lighter grey
TARGET = np.array([1.0, 0.05, 1.0])      # bright yellow: green, almost no UV (unique colour)
DISTRACTOR = np.array([0.75, 0.9, 0.15])  # bright blue-violet: as bright, other colour
N_CH = 3


def _xml(n_distractors):
    rotors = ""
    acts = ""
    for i, (x, y, s) in enumerate([(ARM, ARM, 1), (-ARM, ARM, -1), (-ARM, -ARM, 1), (ARM, -ARM, -1)]):
        rotors += (f'<site name="r{i}" pos="{x} {y} 0.02"/>'
                   f'<geom type="cylinder" pos="{x} {y} 0.03" size="0.055 0.004" rgba="0.2 0.2 0.2 0.5" '
                   f'group="3" contype="0" conaffinity="0" mass="0"/>'
                   f'<geom type="capsule" fromto="0 0 0 {x} {y} 0.02" size="0.008" rgba="0.15 0.15 0.18 1" '
                   f'group="3" contype="0" conaffinity="0" mass="0"/>')
        acts += f'<general site="r{i}" gear="0 0 1 0 0 {s * KM}" ctrllimited="true" ctrlrange="0 {F_MAX}"/>'
    dist = "".join(
        f'<body name="dist{i}" mocap="true" pos="0 0 -5"><geom type="sphere" size="{TARGET_R}" '
        f'rgba="0.35 0.3 0.95 1" group="4" contype="0" conaffinity="0"/></body>' for i in range(n_distractors))
    w = ROOM
    return f"""
<mujoco model="fly_drone">
  <option timestep="{DT}" gravity="0 0 -9.81"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <asset>
    <texture name="fl" type="2d" builtin="checker" rgb1="0.5 0.5 0.5" rgb2="0.38 0.38 0.38" width="64" height="64"/>
    <material name="fl" texture="fl" texrepeat="8 8"/>
    <texture name="wl" type="2d" builtin="checker" rgb1="0.6 0.6 0.62" rgb2="0.45 0.45 0.48" width="64" height="64"/>
    <material name="wl" texture="wl" texrepeat="8 2"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.7 0.8 1" rgb2="1 1 1" width="64" height="64"/>
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="overview" pos="0 -6.8 8.2" xyaxes="1 0 0 0 0.77 0.64" fovy="55"/>
    <geom name="floor" type="plane" size="{w} {w} 0.1" material="fl"/>
    <geom name="wall_e" type="box" pos="{w} 0 {WALL_H / 2}" size="0.05 {w} {WALL_H / 2}" material="wl"/>
    <geom name="wall_w" type="box" pos="{-w} 0 {WALL_H / 2}" size="0.05 {w} {WALL_H / 2}" material="wl"/>
    <geom name="wall_n" type="box" pos="0 {w} {WALL_H / 2}" size="{w} 0.05 {WALL_H / 2}" material="wl"/>
    <geom name="wall_s" type="box" pos="0 {-w} {WALL_H / 2}" size="{w} 0.05 {WALL_H / 2}" material="wl" rgba="1 1 1 0.15"/>
    <body name="target" mocap="true" pos="0 0 1">
      <geom name="target" type="sphere" size="{TARGET_R}" rgba="0.95 0.9 0.1 1" group="4" contype="0" conaffinity="0"/>
    </body>
    {dist}
    <body name="drone" pos="0 0 1">
      <freejoint/>
      <inertial pos="0 0 0" mass="{MASS}" diaginertia="0.003 0.003 0.005"/>
      <geom name="halo" type="sphere" size="0.16" rgba="1 0.45 0.1 0.25" group="5" contype="0" conaffinity="0" mass="0"/>
      <geom name="hull" type="box" size="0.06 0.04 0.025" rgba="0.85 0.3 0.1 1" group="3" mass="0"/>
      <geom name="nose" type="sphere" pos="0.06 0 0.01" size="0.02" rgba="0.1 0.1 0.1 1" group="3" contype="0" conaffinity="0" mass="0"/>
      {rotors}
    </body>
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>"""


# ---------------------------------------------------------------- compound eye
def _fib_sphere(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    phi = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def eye_facets(n_facets):
    """Viewing directions of the RIGHT eye in the drone frame (x forward, y left, z up),
    roughly a fly's field: from 15 deg across the midline (overlap with the other eye)
    to 165 deg behind, from 75 deg below to 80 deg above the horizon. The left eye is
    the mirror image. Returns (directions, azimuth, elevation, spacing in rad)."""
    def inside(v):
        az = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
        el = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
        return (az <= 15) & (az >= -165) & (el >= -75) & (el <= 80)
    lo, hi = n_facets, n_facets * 8
    while lo < hi:                      # smallest sphere sampling with >= n_facets inside
        mid = (lo + hi) // 2
        if inside(_fib_sphere(mid)).sum() >= n_facets:
            hi = mid
        else:
            lo = mid + 1
    v = _fib_sphere(lo)
    v = v[inside(v)][:n_facets]
    spacing = np.sqrt(4 * np.pi / lo)
    az = np.arctan2(v[:, 1], v[:, 0])
    el = np.arcsin(np.clip(v[:, 2], -1, 1))
    return v, az, el, spacing


def _blur_rays(dirs, spacing, n_ring=6):
    """Each facet sees a blurry spot (acceptance angle ~ facet spacing, as in flies):
    1 central ray + a ring of rays at half the spacing."""
    out = [dirs]
    for k in range(n_ring):
        a = 2 * np.pi * k / n_ring
        # two vectors perpendicular to each direction
        ref = np.where(np.abs(dirs[:, 2:3]) < 0.9, [[0, 0, 1.0]], [[1.0, 0, 0]])
        u = np.cross(dirs, ref)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        w = np.cross(dirs, u)
        r = 0.5 * spacing
        d = dirs * np.cos(r) + np.sin(r) * (np.cos(a) * u + np.sin(a) * w)
        out.append(d)
    weights = np.array([0.4] + [0.6 / n_ring] * n_ring)
    return np.concatenate(out, 0), weights


def _quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


class FlyDroneEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, task="reach", n_facets=160, max_steps=300, seed=None, render_mode=None,
                 # robustness (on by default in training: the brain should never see a perfect world)
                 eye_noise=0.01, body_noise=0.05, wind=0.4, gusts=0.2, mass_jitter=0.04,
                 light_jitter=0.2, max_distractors=2, target_visible=True, random_start=0.0, **_):
        assert task in ("reach",), task
        self.task = task
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.eye_noise, self.body_noise = eye_noise, body_noise
        self.wind_max, self.gust_sd, self.mass_jitter = wind, gusts, mass_jitter
        self.light_jitter, self.max_dist = light_jitter, max_distractors
        self.target_visible = target_visible
        self.n_dist_slots = 3
        self.model = mujoco.MjModel.from_xml_string(_xml(self.n_dist_slots))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = FRAME_SKIP
        m = self.model
        self.drone = m.body("drone").id
        self.target_mocap = m.body_mocapid[m.body("target").id]
        self.dist_mocap = [m.body_mocapid[m.body(f"dist{i}").id] for i in range(self.n_dist_slots)]
        self.nominal_mass = float(m.body_mass[self.drone])
        self.inertia = m.body_inertia[self.drone].copy()

        # rotor mixing: [T, tx, ty, tz] = M @ f
        pos = np.array([[ARM, ARM], [-ARM, ARM], [-ARM, -ARM], [ARM, -ARM]])
        spin = np.array([1, -1, 1, -1])
        M = np.stack([np.ones(4), pos[:, 1], -pos[:, 0], spin * KM])
        self.mix_inv = np.linalg.inv(M)

        # compound eye
        self.n_facets = n_facets
        right, az, el, self.spacing = eye_facets(n_facets)
        left = right * np.array([1, -1, 1])
        self.facet_dirs = np.concatenate([right, left])               # (2F, 3), drone frame
        self.facet_az = np.concatenate([az, -az])
        self.facet_el = np.concatenate([el, el])
        self.rays, self.ray_w = _blur_rays(self.facet_dirs, self.spacing)
        self.n_rays = len(self.rays)
        self._gid = np.zeros(self.n_rays, np.int32)
        self._dist = np.zeros(self.n_rays)
        # rays see the room only: drone parts (group 3) and the balls (group 4, added below
        # through each facet's blur profile, so small far balls are still seen faintly)
        self._geomgroup = np.array([1, 1, 1, 0, 0, 0], np.uint8)
        self.sigma = self.spacing / 2.355          # facet blur: width at half height = facet spacing
        g = lambda n: m.geom(n).id  # noqa: E731
        self.floor_id = g("floor")
        self.wall_ids = [g(f"wall_{s}") for s in "ewns"]
        self.target_id = g("target")
        self.dist_ids = [m.body_geomadr[m.body(f"dist{i}").id] for i in range(self.n_dist_slots)]
        self.spectrum = np.zeros((m.ngeom, N_CH))
        self.spectrum[self.floor_id] = FLOOR
        self.spectrum[self.wall_ids] = WALL
        self.spectrum[self.target_id] = TARGET
        self.spectrum[self.dist_ids] = DISTRACTOR

        n_vis = 2 * n_facets * N_CH
        self.n_vis = n_vis
        n_body = 13
        self.observation_space = spaces.Box(-np.inf, np.inf, (2 * n_vis + n_body,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.random_start = random_start
        self.rng = np.random.default_rng(seed)
        self._renderer = None

    # ------------------------------------------------------------------ eye
    def eye_image(self):
        """What the eye sees now: (2 * n_facets, 3) = brightness (R1-6), UV (R7), green (R8)."""
        m, d = self.model, self.data
        R = _quat_to_mat(d.xquat[self.drone])
        world = self.rays @ R.T
        mujoco.mj_multiRay(m, d, d.xpos[self.drone].copy(), world.ravel(), self._geomgroup, 1,
                           self.drone, self._gid, self._dist, None, self.n_rays, 30.0)
        gid, dist = self._gid, self._dist
        val = np.empty((self.n_rays, N_CH))
        hit = gid >= 0
        val[~hit] = SKY * (0.75 + 0.25 * np.clip(world[~hit, 2:3], 0, 1))     # brighter overhead
        val[hit] = self.spectrum[gid[hit]]
        # textures: checkerboard on the floor (0.5 m) and stripes on the walls, so the eye
        # sees its own motion (optic flow), as a fly does over grass or bark
        p = d.xpos[self.drone] + world * dist[:, None]
        fl = gid == self.floor_id
        if fl.any():
            c = (np.floor(p[fl, 0] / self._tile) + np.floor(p[fl, 1] / self._tile)) % 2
            val[fl] *= (0.9 + 0.2 * c)[:, None]
        wl = np.isin(gid, self.wall_ids)
        if wl.any():
            s = np.floor((p[wl, 0] + p[wl, 1]) / 0.4) % 2
            val[wl] *= (0.9 + 0.2 * s)[:, None]
        n_f = len(self.facet_dirs)
        img = (val.reshape(-1, n_f, N_CH) * self.ray_w[:, None, None]).sum(0)
        # balls: blend each into every facet by how much of the facet's blurry view it fills
        eye = d.xpos[self.drone]
        fdir = world[:n_f]
        balls = [(self._target_pos(), TARGET)] if self.target_visible else []
        balls += [(d.mocap_pos[i].copy(), DISTRACTOR) for i in self.dist_mocap if d.mocap_pos[i][2] > 0]
        for c, spec in sorted(balls, key=lambda b: -np.linalg.norm(b[0] - eye)):   # far first
            rel = c - eye
            dist_b = max(np.linalg.norm(rel), TARGET_R * 1.01)
            alpha = np.arcsin(TARGET_R / dist_b)                       # angular radius
            theta = np.arccos(np.clip(fdir @ (rel / dist_b), -1, 1))   # angle facet-ball
            s2 = self.sigma ** 2
            amp = 1 - np.exp(-alpha ** 2 / (2 * s2))
            wgt = amp * np.exp(-theta ** 2 / (2 * (s2 + alpha ** 2 / 4)))
            img = img * (1 - wgt[:, None]) + wgt[:, None] * spec
        return np.clip(img * self._light, 0, 1.5)

    # ------------------------------------------------------------------ helpers
    def _state(self):
        d = self.data
        R = _quat_to_mat(d.xquat[self.drone])
        return d.qpos[:3].copy(), d.qvel[:3].copy(), R, d.qvel[3:6].copy()

    def _heading(self, R):
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def _target_pos(self):
        return self.data.mocap_pos[self.target_mocap].copy()

    def _place_mocap(self, idx, p):
        self.data.mocap_pos[idx] = p

    def _random_point(self, margin=0.8, z=(0.5, 2.0)):
        r = ROOM - margin
        return np.array([self.rng.uniform(-r, r), self.rng.uniform(-r, r), self.rng.uniform(*z)])

    def _get_obs(self):
        img = self.eye_image()
        if self.eye_noise > 0:
            img = img + self.rng.normal(0, self.eye_noise, img.shape)
        img = img.ravel()
        d_img = img - (self._prev_img if self._prev_img is not None else img)
        self._prev_img = img
        p, v, R, w = self._state()
        vb, wb = R.T @ v, w.copy()
        if self.body_noise > 0:
            vb = vb + self.rng.normal(0, self.body_noise, 3)
            wb = wb + self.rng.normal(0, self.body_noise, 3)
        body = np.concatenate([vb, R.T @ np.array([0, 0, -1.0]), wb, self._last_action])
        return np.concatenate([img, d_img, body]).astype(np.float32)

    # ------------------------------------------------------------------ stabiliser
    def _stabilise(self):
        """Haltere-like reflex: hold the commanded velocity (v_ref) and heading."""
        d = self.data
        p, v, R, w = self._state()
        acc = 3.0 * (self._v_ref - v) + self._acc_ff
        h = np.linalg.norm(acc[:2])
        if h > 3.5:
            acc[:2] *= 3.5 / h
        acc[2] = np.clip(acc[2], -4.0, 4.0)
        f = self.nominal_mass * (acc + np.array([0, 0, 9.81]))
        fh, fz = np.linalg.norm(f[:2]), max(f[2], 0.5)
        if fh > fz * np.tan(TILT_MAX):
            f[:2] *= fz * np.tan(TILT_MAX) / fh
        f[2] = fz
        zb = f / np.linalg.norm(f)
        xc = np.array([np.cos(self._yaw_ref), np.sin(self._yaw_ref), 0.0])
        yb = np.cross(zb, xc)
        yb /= np.linalg.norm(yb)
        xb = np.cross(yb, zb)
        Rd = np.stack([xb, yb, zb], 1)
        E = Rd.T @ R - R.T @ Rd
        e_R = 0.5 * np.array([E[2, 1], E[0, 2], E[1, 0]])
        w_d = R.T @ Rd @ np.array([0, 0, self._yaw_rate])
        tau = self.inertia * (-150.0 * e_R - 20.0 * (w - w_d))
        T = float(f @ R[:, 2])
        d.ctrl[:] = np.clip(self.mix_inv @ np.array([T, *tau]), 0, F_MAX)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        m.body_mass[self.drone] = self.nominal_mass * (1 + self.rng.uniform(-self.mass_jitter, self.mass_jitter))
        self._light = 1 + self.rng.uniform(-self.light_jitter, self.light_jitter)
        self._tile = self.rng.uniform(0.35, 0.7)
        ang = self.rng.uniform(0, 2 * np.pi)
        self._wind = self.rng.uniform(0, self.wind_max) * np.array([np.cos(ang), np.sin(ang), 0.0])
        self._gust = np.zeros(3)

        start = self._random_point(margin=1.0, z=(0.8, 1.8))
        for _ in range(100):
            target = self._random_point(margin=0.8, z=(0.5, 2.0))
            if 1.5 <= np.linalg.norm(target - start) <= 4.0:
                break
        if self.random_start > 0 and self.rng.random() < self.random_start:
            # curriculum: start close to the target (still has to see it and stop on it)
            off = self.rng.normal(size=3)
            off *= self.rng.uniform(0.5, 1.5) / np.linalg.norm(off)
            start = np.clip(target + off, [-ROOM + 1, -ROOM + 1, 0.6], [ROOM - 1, ROOM - 1, 2.0])
        yaw = self.rng.uniform(-np.pi, np.pi)
        d.qpos[:3] = start
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self._place_mocap(self.target_mocap, target)
        n_d = self.rng.integers(0, self.max_dist + 1) if self.max_dist > 0 else 0
        for i, idx in enumerate(self.dist_mocap):
            p = np.array([0, 0, -5.0])
            if i < n_d:
                for _ in range(100):
                    p = self._random_point(margin=0.6, z=(0.5, 2.0))
                    if np.linalg.norm(p - target) > 1.0 and np.linalg.norm(p - start) > 0.8:
                        break
            self._place_mocap(idx, p)
        mujoco.mj_forward(m, d)

        self._yaw_ref, self._yaw_rate = yaw, 0.0
        self._v_ref, self._acc_ff = np.zeros(3), np.zeros(3)
        self._last_action = np.zeros(4)
        self._steps, self._hold = 0, 0
        self._prev_img = None
        self._phi = self._potential()
        return self._get_obs(), {}

    def _target_dist(self):
        return float(np.linalg.norm(self.data.qpos[:3] - self._target_pos()))

    def _potential(self):
        return -PROGRESS * self._target_dist() + HOLD_POT * self._hold

    def _command(self, a):
        """Brain command -> smooth velocity / turn-rate references for the stabiliser."""
        _, _, R, _ = self._state()
        yaw = self._heading(R)
        c, s = np.cos(yaw), np.sin(yaw)
        v_cmd = np.array([c * a[0] * V_XY - s * a[1] * V_XY, s * a[0] * V_XY + c * a[1] * V_XY, a[2] * V_Z])
        sp = np.linalg.norm(v_cmd[:2])
        if sp > V_XY:
            v_cmd[:2] *= V_XY / sp
        dt = DT * FRAME_SKIP
        dv = v_cmd - self._v_ref
        n = np.linalg.norm(dv)
        if n > A_MAX * dt:
            dv *= A_MAX * dt / n
        self._v_ref = self._v_ref + dv
        self._acc_ff = dv / dt
        self._yaw_rate += np.clip(a[3] * YAW_RATE - self._yaw_rate, -YAW_ACC * dt, YAW_ACC * dt)

    def _crashed(self):
        p, _, R, _ = self._state()
        return bool(p[2] < 0.12 or p[2] > 3.0 or abs(p[0]) > ROOM - 0.2 or abs(p[1]) > ROOM - 0.2
                    or R[2, 2] < np.cos(np.deg2rad(70)))

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        self._command(a)
        d = self.data
        for _ in range(FRAME_SKIP):
            self._yaw_ref += self._yaw_rate * DT
            self._gust += -self._gust * DT / 1.0 + self.gust_sd * np.sqrt(2 * DT) * self.rng.normal(size=3) * np.array([1, 1, 0.3])
            d.xfrc_applied[self.drone, :3] = self.model.body_mass[self.drone] * (self._wind + self._gust)
            self._stabilise()
            mujoco.mj_step(self.model, d)
        self._steps += 1

        crashed = self._crashed()
        dist = self._target_dist()
        speed = float(np.linalg.norm(d.qvel[:3]))
        on = dist < SOLVE_DIST and speed < SOLVE_SPEED
        self._hold = self._hold + 1 if on else 0
        success = self._hold >= HOLD_STEPS

        phi = self._potential()
        r = phi - self._phi
        self._phi = phi
        r -= TIME_COST + JERK_COST * float(np.sum((a - self._last_action) ** 2))
        if success:
            r += SUCCESS
        if crashed:
            r += CRASH
        self._last_action = a
        terminated = success or crashed
        truncated = self._steps >= self.max_steps
        info = {"is_success": success, "crashed": crashed, "dist": dist}
        return self._get_obs(), float(r), terminated, truncated, info

    # ------------------------------------------------------------------ rendering
    def eye_view(self, size=190):
        """The two eyes as images (azimuth across, elevation up). Colours: red = brightness
        (R1-6), green = green (R8), blue = ultraviolet (R7)."""
        img = self.eye_image()
        F = self.n_facets
        h = size // 2
        out = np.zeros((h + 4, 2 * size + 6, 3))
        for e, x0 in ((1, 0), (0, size + 6)):           # left eye on the left
            sl = slice(e * F, (e + 1) * F)
            az, el, val = self.facet_az[sl], self.facet_el[sl], img[sl]
            # map each eye's own azimuth range onto its panel
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
            self._vopt.geomgroup[:] = 1                      # show drone, balls and halo
        self._renderer.update_scene(self.data, camera="overview", scene_option=self._vopt)
        frame = self._renderer.render().copy()
        eye = self.eye_view()
        frame[6:6 + eye.shape[0], 6:6 + eye.shape[1]] = eye
        return frame

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def drone_controller(env, obs=None):
    """Hand-written solution using the TRUE target position (the brain never gets it):
    turn towards the target, fly to it, slow down near it."""
    p, v, R, _ = env._state()
    rel = env._target_pos() - p
    yaw = env._heading(R)
    c, s = np.cos(yaw), np.sin(yaw)
    fwd = c * rel[0] + s * rel[1]
    left = -s * rel[0] + c * rel[1]
    k = 1.2
    a = np.array([k * fwd / V_XY, k * left / V_XY, k * rel[2] / V_Z,
                  2.0 * np.arctan2(left, fwd) / YAW_RATE])
    return np.clip(a, -1, 1)


class EyeOnlyPilot:
    """Hand-written pilot that uses ONLY the observation (eye image + body sense), never
    the true positions. Proves the eye carries enough information for the task.
    Finds the facet that looks most like the ball (green, almost no UV), keeps a short
    memory of that direction, turns and flies towards it, slows down as the ball grows
    in view. Searches by wandering and turning when nothing is seen.
    Solves ~90 % of rooms (also with 3 decoy balls)."""

    def __init__(self, env):
        self.e, self.mem, self.conf = env, None, 0

    def __call__(self, env, obs):
        e = self.e
        F2 = 2 * e.n_facets
        img = obs[:F2 * N_CH].reshape(F2, N_CH)
        sc = img[:, 2] - 1.5 * img[:, 1] - 0.3 * img[:, 0]
        k = int(np.argmax(sc))
        size = int((sc > 0.15).sum())
        if sc[k] > 0.0:
            d = e.facet_dirs[k]
            self.mem = d if self.mem is None else 0.6 * self.mem + 0.4 * d
            self.mem = self.mem / np.linalg.norm(self.mem)
            self.conf = 10
        elif self.conf > 0:
            self.conf -= 1
        else:
            self.mem = None
        if self.mem is None:
            return np.array([0.35, 0, 0, 0.6])
        d = self.mem
        yaw = np.arctan2(d[1], d[0])
        near = min(size / 45, 1.0)
        fwd = max(1 - near, 0.15) * max(d[0], 0) if abs(yaw) < 0.6 else 0
        return np.clip(np.array([fwd, 1.5 * d[1] * (1 - near) + 0.8 * d[1] * near, 2.0 * d[2],
                                 2.5 * yaw / YAW_RATE]), -1, 1)
