"""Waste-sorting station (`panda_waste`): a fixed Franka Panda with a suction cup over a conveyor belt, as on
recycling lines (AMP Robotics, ZenRobotics). One item per episode lies on the belt; the arm picks it by suction
and drops it into the bin of its material: plastic, metal, paper or glass (four bins beside the belt).

Two senses, each only partly sufficient, so the task needs both:
  - the fly's compound eyes (head eye over the station, wrist eye on the hand; photoreceptors R1-6, R7 UV, R8
    green): item colour, UV reflectance and apparent size. Look-alikes across materials on purpose: clear PET
    vs clear glass, green plastic vs green glass vs green-painted cans, brown cardboard vs brown glass; paper's
    optical brighteners glow in UV, PET absorbs it. Labels and dirt change the look.
  - a close-range material sensor (a stand-in for the NIR spectrometers of real sorting lines), delivered
    through the fly's smell pathway: each material is a fixed mixture of the 8 odorants with measured receptor
    responses (DoOR), sensed at the two "antennae" beside the cup and fading with distance. Residue and
    composite items (e.g. a paper-wrapped can) mix in another material's reading.

Suction, as on industrial vacuum grippers: the cup grips whatever it touches (a weld between hand and item
switches on at contact), and the policy's fourth command is the release valve: > 0 lets go (and keeps the cup
from gripping again while it stays > 0). So the one gripper decision left is the sorting decision: where to
let go. Success: the item released and resting in the right bin for 0.2 s. A
release into a wrong bin ends the episode. Tests (no retraining): eyes_off / smell_off (a failed sense),
light changes, item_types="heldout" (two item types never shown in training: HELDOUT_ITEMS).
"""
from __future__ import annotations

import mujoco
import numpy as np

from .panda import (BIN_FLOOR, FRAME_SKIP, LIFT_HOLD, SMELL_ANTENNAE, SMELL_FLICKER, SMELL_NOISE, SMELL_SIGMA, STEP,
                    WS_HI, WS_LO, FlyPandaEnv)

MATERIALS = ("plastic", "metal", "paper", "glass")
WASTE_BINS = np.array([[0.62, 0.24], [0.62, 0.08], [0.62, -0.08], [0.62, -0.24]])   # in MATERIALS order
PICK_LO, PICK_HI = np.array([0.36, -0.17]), np.array([0.46, 0.17])                  # where items lie on the belt
# item types: (name, material, photoreceptor view (R1-6, R7 UV, R8 green), render colour, half sizes x y z (m))
ITEMS = [
    ("clear PET bottle", 0, [0.70, 0.15, 0.70], [0.85, 0.90, 0.95, 1], [0.040, 0.018, 0.018]),
    ("green PET bottle", 0, [0.40, 0.10, 0.65], [0.20, 0.60, 0.30, 1], [0.040, 0.018, 0.018]),
    ("white HDPE tub", 0, [0.85, 0.30, 0.80], [0.95, 0.95, 0.92, 1], [0.030, 0.030, 0.020]),
    ("aluminium can", 1, [0.75, 0.55, 0.70], [0.75, 0.77, 0.80, 1], [0.032, 0.020, 0.020]),
    ("red painted can", 1, [0.30, 0.25, 0.15], [0.80, 0.10, 0.10, 1], [0.032, 0.020, 0.020]),
    ("green painted can", 1, [0.40, 0.25, 0.62], [0.15, 0.55, 0.25, 1], [0.032, 0.020, 0.020]),
    ("white paper", 2, [0.90, 0.75, 0.85], [0.97, 0.97, 0.97, 1], [0.040, 0.030, 0.003]),
    ("cardboard", 2, [0.45, 0.18, 0.35], [0.60, 0.45, 0.28, 1], [0.040, 0.030, 0.012]),
    ("clear glass jar", 3, [0.68, 0.12, 0.68], [0.80, 0.88, 0.90, 1], [0.025, 0.025, 0.030]),
    ("green glass bottle", 3, [0.38, 0.08, 0.62], [0.10, 0.45, 0.20, 1], [0.040, 0.018, 0.018]),
    ("brown glass bottle", 3, [0.25, 0.05, 0.18], [0.35, 0.20, 0.08, 1], [0.040, 0.018, 0.018]),
]
# two item types are never shown in training, only in a test: a white HDPE tub (looks like paper, except in UV)
# and a green painted can (looks like green bottles). Sorting them right needs the material sensor, not the look.
HELDOUT_ITEMS = (2, 5)
TRAIN_ITEMS = tuple(k for k in range(len(ITEMS)) if k not in HELDOUT_ITEMS)
ITEM_MASS = 0.05                               # kg, every item (the model's compiled mass: changing it at run time left
                                               # the constraint scaling stale, and 0.1-0.2 kg items blew up when held)
LOOK_JITTER = 0.20                             # each receptor channel x U(1 -+ this) per item (lighting, wear)
LABEL_P, LABEL_W = 0.4, (0.25, 0.5)            # share of items with a label / dirt, and its weight in the look
MIX_P, MIX_W = 0.35, (0.3, 0.7)               # share of items with residue / composite: another material's reading
CARRY_Z = 0.12                                 # teacher: carry height of the cup (m)
DROP_Z = 0.015                                 # teacher: let go with the item's underside this high above the bin floor
WS_LO_W = np.array([0.30, -0.32, 0.002])       # the cup may go down to the belt (thin paper is 6 mm high)
WRONG_BIN_REWARD = -1.0
SORT_REWARD = 2.0                              # the item sorted into the right bin (the episode ends)
OVER_Z = 0.08                                  # reward: carry towards this height above the bin floor (rims: 4 cm)


class WasteSortEnv(FlyPandaEnv):
    def __init__(self, eyes_off=False, item_types=None, **kw):
        kw.pop("task", None)
        super().__init__(task="waste", **kw)
        m = self.model
        self.cup_geom = m.geom("cup").id
        self._cup_col = (int(m.geom_contype[self.cup_geom]), int(m.geom_conaffinity[self.cup_geom]))
        self.suction_eq = m.equality("suction").id
        self.hand_body, self.item_body = m.body("hand").id, m.body("block").id
        self.eyes_off = bool(eyes_off)                 # test: both eyes dark (a failed camera)
        if item_types == "heldout":
            item_types = HELDOUT_ITEMS
        elif item_types == "all":
            item_types = range(len(ITEMS))
        self.item_types = list(item_types) if item_types is not None else list(TRAIN_ITEMS)
        self._held, self._suck = False, False
        self._item, self._mat = 0, 0
        self._mat_drive = np.zeros(self.smell_resp.shape[0])
        self._wrong = False

    # ------------------------------------------------------------------ episode
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        m, d = self.model, self.data
        self._detach()
        self._suck, self._wrong = True, False
        for mc, c in zip(self.bin_mocaps, WASTE_BINS):
            d.mocap_pos[mc] = [c[0], c[1], 0.0]
        k = int(self.rng.choice(self.item_types))
        name, mat, look, rgba, half = ITEMS[k]
        self._item, self._mat = k, mat
        self.bin_mocap = self.bin_mocaps[mat]
        half = np.array(half) * self.rng.uniform(0.9, 1.1, 3)
        g, b = self.block_geom, self.item_body
        m.geom_size[g] = half
        m.geom_rbound[g] = float(np.linalg.norm(half))
        m.geom_aabb[g] = [0, 0, 0, *half]
        m.geom_rgba[g] = rgba
        mass = ITEM_MASS
        m.body_mass[b] = mass
        m.body_inertia[b] = mass / 3 * np.array([half[1] ** 2 + half[2] ** 2, half[0] ** 2 + half[2] ** 2,
                                                 half[0] ** 2 + half[1] ** 2])
        spec = np.array(look) * self.rng.uniform(1 - LOOK_JITTER, 1 + LOOK_JITTER, 3)
        if self.rng.random() < LABEL_P:                # a label or dirt: part of the view is some other colour
            w = self.rng.uniform(*LABEL_W)
            spec = (1 - w) * spec + w * self.rng.uniform(0.1, 0.9, 3)
        self._block_spec = spec
        self._obj_r = float(max(half[0], half[1]))     # the eye sees it as a blob about this size
        self.rest_z = BIN_FLOOR + half[2]
        drive = self.smell_drive[mat]
        if self.rng.random() < MIX_P:                  # residue / composite: some of another material's reading
            other = int(self.rng.choice([j for j in range(len(MATERIALS)) if j != mat]))
            w = self.rng.uniform(*MIX_W)
            drive = (1 - w) * drive + w * self.smell_drive[other]
        self._mat_drive = drive
        xy = self.rng.uniform(PICK_LO, PICK_HI)
        yaw = self.rng.uniform(-np.pi, np.pi)
        qa, da = self.block_q, m.jnt_dofadr[m.body("block").jntadr[0]]
        d.qpos[qa:qa + 7] = [xy[0], xy[1], half[2] + 0.001, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qvel[da:da + 6] = 0
        mujoco.mj_forward(m, d)
        for _ in range(30):
            mujoco.mj_step(m, d)
        if self.grasp_start > 0 and self.rng.random() < self.grasp_start:
            # training only (curriculum, as in the lift / place tasks): start with the item already held, the hand
            # somewhere between the belt and the bins, so carrying and dropping are found too. Without it, PPO
            # learned to suck the item up and then sat on the belt holding it (700k steps, 0 % sorted).
            tcp = np.r_[self.rng.uniform([0.38, -0.22], [0.58, 0.22]), self.rng.uniform(0.04, 0.16)]
            q = self._ik_solve(tcp, self._home, iters=60)
            d.qpos[:7] = d.ctrl[:7] = q
            mujoco.mj_forward(m, d)
            tcp = self.tcp_pos()
            d.qpos[qa:qa + 3] = tcp - np.array([0, 0, half[2] + 0.001])
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
            self._attach()
            self._suck = True
            self._last_action = np.array([0, 0, 0, -1.0])
            for _ in range(10):
                mujoco.mj_step(m, d)
        self._hold, self._steps = 0, 0
        self._prev_img, self._adapt = None, None
        self._phi_prev = None
        self._cmd = np.zeros(3)
        if not self._held:
            self._last_action = np.zeros(4)
        return self._get_obs(), {"item": name, "material": MATERIALS[mat], "held_start": self._held}

    @property
    def _grip_ctrl(self):
        """For code written for the finger gripper (imitation's gripper-decision weighting): gripping = a closed
        hand (0), venting = an open one (255)."""
        return 0.0 if getattr(self, "_suck", True) else 255.0

    @_grip_ctrl.setter
    def _grip_ctrl(self, v):
        pass                                           # (the base class sets the finger target; the cup ignores it)

    # ------------------------------------------------------------------ senses
    def cup_touching(self):
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            if {c.geom1, c.geom2} == {self.cup_geom, self.block_geom}:
                return True
        return False

    def block_touched(self):
        return self.cup_touching()

    def is_grasped(self):
        return self._held

    def smell_obs(self):
        """The material sensor's reading through the two antennae (fades with distance from the item)."""
        if self.smell_off:
            return np.zeros(self.n_smell)
        tcp, it = self.tcp_pos(), self.block_pos()
        out = []
        for a in SMELL_ANTENNAE:
            near = np.exp(-float(np.sum((tcp + a - it) ** 2)) / (2 * SMELL_SIGMA ** 2))
            conc = near * np.exp(self.rng.normal(0, SMELL_FLICKER)) + 0.02
            drive = conc * self._mat_drive
            out.append(np.tanh(2.0 * drive) + self.rng.normal(0, SMELL_NOISE * self.smell_noise_x, drive.shape))
        return np.concatenate(out)

    def _get_obs(self):
        img = self.eye_image()
        if self.eye_noise > 0 and not self.eye_adapt:
            img = img + self.rng.normal(0, self.eye_noise, img.shape)
        img = img.ravel()
        if self.eyes_off:
            img = np.zeros_like(img)
        d_img = img - (self._prev_img if self._prev_img is not None else img)
        self._prev_img = img
        p, v = self.tcp_pos(), self.tcp_vel()
        if self.body_noise > 0:
            p = p + self.rng.normal(0, self.body_noise, 3)
            v = v + self.rng.normal(0, 2 * self.body_noise, 3)
        # body sense, same 13-value layout as the gripper tasks: hand position, velocity, suction on, holding,
        # cup touching, last command
        body = np.concatenate([(p - 0.5 * (WS_LO + WS_HI)) * 4, v * 2, [1.0 if self._suck else -1.0],
                               [float(self._held)], [float(self.cup_touching())], self._last_action])
        self._last_smell = self.smell_obs()
        out = [img, d_img, body, self._last_smell]
        if self.privileged:
            out.append(self.privileged_state())
        return np.concatenate(out).astype(np.float32)

    def privileged_state(self):
        """True state for the training-only critic (never given to the policy): 25 values."""
        tcp, it, c = self.tcp_pos(), self.block_pos(), self.bin_pos()
        top = it + np.array([0, 0, self.model.geom_size[self.block_geom][2]])
        return np.concatenate([tcp, it, top - tcp, c, it - (c + np.array([0, 0, self.rest_z])),
                               [float(self._held), float(self.cup_touching()), float(self._suck)],
                               np.eye(len(MATERIALS))[self._mat], self.model.geom_size[self.block_geom]])

    # ------------------------------------------------------------------ suction
    def _attach(self):
        m, d = self.model, self.data
        ph, qh = d.xpos[self.hand_body], d.xquat[self.hand_body]
        pi, qi = d.xpos[self.item_body], d.xquat[self.item_body]
        qh_inv = np.zeros(4)
        mujoco.mju_negQuat(qh_inv, qh)
        rel_p = np.zeros(3)
        mujoco.mju_rotVecQuat(rel_p, pi - ph, qh_inv)
        rel_q = np.zeros(4)
        mujoco.mju_mulQuat(rel_q, qh_inv, qi)
        e = self.suction_eq
        m.eq_data[e][:3] = 0.0                         # anchor: the item's origin
        m.eq_data[e][3:6] = rel_p                      # where the item sits in the hand frame now
        m.eq_data[e][6:10] = rel_q
        m.eq_data[e][10] = 1.0
        d.eq_active[e] = 1
        # while held, the cup does not collide (the weld holds the item where it was pressed on; cup contact
        # pushing it out at the same time made heavy items blow up the physics)
        m.geom_contype[self.cup_geom] = m.geom_conaffinity[self.cup_geom] = 0
        self._held = True

    def _detach(self):
        self.data.eq_active[self.suction_eq] = 0
        self.model.geom_contype[self.cup_geom], self.model.geom_conaffinity[self.cup_geom] = self._cup_col
        self._held = False

    # ------------------------------------------------------------------ step
    def step(self, action):
        a = np.clip(np.nan_to_num(np.asarray(action, dtype=np.float64)), -1, 1)
        m, d = self.model, self.data
        tcp = self.tcp_pos()
        self._cmd = self._cmd + self.cmd_smooth * (a[:3] - self._cmd)
        self._target = np.clip(self._target + self._cmd * STEP, WS_LO_W, WS_HI)
        off = self._target - tcp
        n = np.linalg.norm(off)
        if n > 0.05:
            self._target = tcp + off * 0.05 / n
        d.ctrl[:7] = self._ik_solve(self._target, d.qpos[:7].copy())
        d.ctrl[7] = 255.0                              # (hidden fingers stay open)
        self._suck = bool(a[3] <= 0)                   # the release valve: a[3] > 0 vents the cup
        if not self._suck and self._held:
            self._detach()
        bad0 = self._bad_count()
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(m, d)
            if self._suck and not self._held and self.cup_touching():
                self._attach()
        self._steps += 1
        self._last_action = a
        phi, success, d_t = self._reward()
        # reward paid once, when progress is made (the change of the staged measure: potential-based shaping, Ng
        # et al. 1999), plus SORT_REWARD for the right bin; a wrong bin costs WRONG_BIN_REWARD. Paid per step
        # instead, holding the item over a bin earned 0.75 a step, nearly what sorting earns, and dropping into a
        # wrong bin ended it: PPO learned to hold forever (0 % sorted, 700k steps, small and large MLP alike)
        r = phi - (self._phi_prev if self._phi_prev is not None else phi)
        self._phi_prev = phi
        truncated = self._steps >= self.max_steps
        unstable = self._bad_count() > bad0 or not np.isfinite(d.qpos).all()
        terminated = False
        wrong = (not self._held) and self.in_wrong_bin()
        if unstable:
            r, success, truncated = 0.0, False, True
        elif wrong:
            r, terminated, self._wrong = WRONG_BIN_REWARD, True, True
        elif success:
            r, terminated = r + SORT_REWARD, True
        info = {"is_success": bool(success), "dist": d_t, "sim_unstable": unstable, "wrong_bin": bool(wrong),
                "held": self._held, "in_bin": self.in_bin(), "item": ITEMS[self._item][0],
                "material": MATERIALS[self._mat]}
        obs = self._get_obs()
        if unstable:
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs, float(np.nan_to_num(r)), terminated, truncated and not terminated, info

    def _reward(self):
        """Staged, every stage pays more than the one before: reach the item's top (0..1) -> held (+1) -> carried to
        a point above its bin (+2 x closeness) -> lowered into it (+0.5) -> in the bin, released (5) -> stays 0.2 s:
        success (6)."""
        tcp, it, c = self.tcp_pos(), self.block_pos(), self.bin_pos()
        top = it + np.array([0, 0, self.model.geom_size[self.block_geom][2]])
        d_t = float(np.linalg.norm(tcp - top))
        dxy = float(np.linalg.norm(it[:2] - c[:2]))
        r = 1 - np.tanh(5 * d_t)
        if self._held:
            # carried towards a point ABOVE its bin (tanh(3 x): a clear pull from 30 cm away). Paid by horizontal
            # distance only, the item was dragged along the table and stuck on the 4 cm rim; over the bin, the
            # height no longer counts and lowering it in pays
            over = c + np.array([0, 0, self.rest_z + OVER_Z])
            d_over = float(np.linalg.norm(it - over)) if dxy >= 0.03 else 0.0
            r += 1.0 + 2.0 * (1 - np.tanh(3 * d_over))
            if dxy < 0.03:
                r += 0.5 * (1 - np.tanh(10 * max(0.0, it[2] - self.rest_z - DROP_Z)))   # lowered into it
        released = self.in_bin() and not self._held
        if released:
            r = 5.0
        self._hold = self._hold + 1 if released else 0
        success = self._hold >= LIFT_HOLD
        if success:
            r = 6.0
        return r / 6.0, success, d_t


def waste_teacher(env):
    """Scripted sorter (TRUE item position and material; for checks and imitation only): above the item, down
    onto its top (the cup grips), up to carry height, over the right bin, down to DROP_Z, release (a[3] > 0 vents
    the cup; kept venting while moving away, so it does not grip the item again)."""
    tcp, it = env.tcp_pos(), env.block_pos()
    a = np.zeros(4)
    hz = float(env.model.geom_size[env.block_geom][2])
    if env.is_grasped():
        c = env.bin_pos()
        a[3] = -1.0
        dxy = float(np.linalg.norm(tcp[:2] - c[:2]))
        if dxy > 0.015:
            want = np.array([tcp[0], tcp[1], CARRY_Z]) if tcp[2] < CARRY_Z - 0.03 and dxy > 0.06 else \
                np.array([c[0], c[1], CARRY_Z])
        else:
            want = np.array([c[0], c[1], env.rest_z + hz + DROP_Z])     # (the cup sits on the item's top)
            if tcp[2] < env.rest_z + hz + DROP_Z + 0.008:
                a[3] = 1.0                             # over the bin, low enough: let go
        target = want
    elif env.in_bin() or env.in_wrong_bin():
        target = np.array([tcp[0], tcp[1], CARRY_Z])    # done: up and away
        a[3] = 1.0
    else:
        top = it + np.array([0, 0, hz])
        xy_err = float(np.linalg.norm((top - tcp)[:2]))
        if xy_err > 0.01:
            target = np.array([top[0], top[1], max(top[2] + 0.05, min(tcp[2], CARRY_Z))])
            a[3] = 1.0
        else:
            target = top - np.array([0, 0, 0.004])      # press onto the top, suction on
            a[3] = -1.0
    a[:3] = np.clip((target - tcp) / STEP * 0.6, -1, 1)
    return a
