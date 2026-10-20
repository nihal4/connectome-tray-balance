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
    lift    grasp the yellow block between the two fingers and lift it 8 cm off the table,
            holding it there for 0.2 s (100 steps = 4 s). Same eyes, decoys and body sense,
            plus TOUCH: the brain feels whether each finger presses on something.
    place   pick and place: grasp the yellow block, carry it into a shallow red bin (new place
            every episode, seen by the eyes) and let go; success = block resting inside the bin,
            released, for 0.2 s (150 steps = 6 s).
    sort    sort by colour, like a sorting station: ONE block per episode, yellow, blue or orange at
            random, on the near half of the table; three identical red bins in a row along the far
            side (left, centre, right, a little jitter). Yellow goes left, blue centre, orange right.
            No decoys. Success as in place, but only in the block's own bin (150 steps).
    smell   ripe or rotten, by smell alone: ONE fruit (the yellow block) per episode, ripe or rotten at
            random, looking exactly the same; two identical bins, ripe -> left, rotten -> right. The
            brain smells with two antennae beside the gripper (fly olfactory receptor responses from DoOR,
            data/smell_door.npz). Success as in sort (150 steps).

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
# task (lift), ManiSkill PickCube-style staged reward: reach the block (1 - tanh(5 d)),
# +1 while it is held between both fingers, +2 x lift progress while held, success = 5.
# Lift only counts while held, so pushing or flicking the block earns nothing extra.
LIFT_GOAL = 0.08             # raise the block's centre this far (m) ...
LIFT_HOLD = 5                # ... while held, for 5 steps (0.2 s)
# lift_reward="goal" (ManiSkill PickCube style): while held, reward the block for being near a
# point in the air LIFT_GOAL straight above where it started (2 x (1 - tanh(10 d))), so going
# up pays and carrying it sideways costs. Success = held within GOAL_DIST of that point, 0.2 s.
GOAL_DIST = 0.025
N_PRIV = 17                  # size of privileged_state() (reach, lift)
# task (place): a shallow open box, 14 x 14 cm outside, 1 cm floor, 1.5 cm high rim
BIN_HALF = 0.07              # outer half size
BIN_WALL = 0.01              # rim thickness
BIN_FLOOR = 0.01             # floor thickness (its top is the block's resting surface)
BIN_RIM_H = 0.015            # rim height above the floor
BIN_IN = 0.045               # block centre within this of the bin centre (x and y) = inside
BIN_LO = np.array([0.38, -0.19])   # where bins appear
BIN_HI = np.array([0.62, 0.19])
BIN_MIN_GAP = 0.15           # block starts at least this far from the bin centre
PLACE_CARRY = 0.08           # teacher: carry the block this high above the bin floor
REST_Z = BIN_FLOOR + BLOCK_H # block centre height when resting in the bin


def n_priv(task):
    """Size of privileged_state() for a task (place adds the bin position and block-to-bin;
    sort also the block's colour, one-hot)."""
    return N_PRIV + {"place": 6, "sort": 9, "smell": 8}.get(task, 0)


SORT_BINS = np.array([[0.60, 0.17], [0.60, 0.0], [0.60, -0.17]])   # sort: left, centre, right bin
SORT_JITTER = 0.015          # sort: each bin moves up to this much per episode
SORT_BLOCK_LO = np.array([0.35, -0.22])   # sort: blocks appear on the near half
SORT_BLOCK_HI = np.array([0.47, 0.22])
# smell: the fly's olfactory receptor responses to each fruit (one value per glomerulus type), two
# antennae beside the gripper. Concentration falls off with distance (Gaussian, SMELL_SIGMA), with
# turbulence-like flicker (log-normal per step) and a faint background; receptor output saturates.
SMELL_TABLE = os.path.join(ROOT, "data", "smell_door.npz")
SMELL_SIGMA = 0.10           # m
SMELL_ANTENNAE = np.array([[0.0, 0.04, 0.06], [0.0, -0.04, 0.06]])   # left, right: beside / above the TCP
SMELL_FLICKER = 0.25         # log-normal sd of the concentration, per step and antenna
SMELL_NOISE = 0.03           # receptor noise
SMELL_BINS = SORT_BINS[[0, 2]]   # ripe -> left bin, rotten -> right bin


# what each photoreceptor type sees: (R1-6 brightness, R7 UV, R8 green)
TABLE = np.array([0.35, 0.30, 0.35])     # grey
ROBOT = np.array([0.80, 0.70, 0.80])     # white-grey robot
WALLS = np.array([0.55, 0.60, 0.50])     # background (no hit): pale, a bit UV
TARGET = np.array([1.0, 0.05, 1.0])      # yellow block: green, almost no UV (unique)
DECOY = np.array([0.75, 0.90, 0.15])     # blue-violet decoy block
# eye_adapt: each receptor type scales its output so its average over the eye is ADAPT_REF (about the
# average under normal light without adaptation), moving ADAPT_RATE of the way per step (25 Hz)
ADAPT_REF = np.array([0.45, 0.45, 0.45])
ADAPT_RATE = 0.3
BIN = np.array([0.20, 0.65, 0.05])       # red bin: dark, some UV, no green (unlike table, block, decoys)
# sort: block colours (photoreceptor view: brightness, UV, green; render colour), and their bins.
# The bins all look the same (red, as in place), so a bin never looks like a block: where each
# colour goes is the rule to learn. Orange = bright, no UV, little green (unlike yellow, blue, bins).
SORT_COLOURS = [("yellow", TARGET, [0.95, 0.85, 0.1, 1]),     # -> left bin
                ("blue", DECOY, [0.3, 0.3, 0.95, 1]),         # -> centre bin
                ("orange", np.array([0.95, 0.05, 0.2]), [0.95, 0.5, 0.1, 1])]   # -> right bin


def _turn(u, v):
    """Quaternion of the smallest rotation that turns direction u into direction v."""
    ax = np.cross(u, v)
    s = np.linalg.norm(ax)
    ang = np.arctan2(s, float(u @ v))
    ax = ax / s if s > 1e-9 else np.array([1.0, 0, 0])
    return np.array([np.cos(ang / 2), *(np.sin(ang / 2) * ax)])


def _build_model(n_decoys, n_facets, bins=()):
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
    for k, (bname, rgba) in enumerate(bins):     # shallow boxes the eyes see by ray (group 0), moved per episode
        bn = wb.add_body()
        bn.name, bn.mocap, bn.pos = bname, True, [0.5, 0.2 - 0.2 * k, 0]
        h, w, f, rim = BIN_HALF, BIN_WALL, BIN_FLOOR, BIN_RIM_H
        parts = [("floor", [h, h, f / 2], [0, 0, f / 2])]
        zc, hz = f + rim / 2, rim / 2
        parts += [("n", [h, w / 2, hz], [0, h - w / 2, zc]), ("s", [h, w / 2, hz], [0, -h + w / 2, zc]),
                  ("e", [w / 2, h - w, hz], [h - w / 2, 0, zc]), ("w", [w / 2, h - w, hz], [-h + w / 2, 0, zc])]
        for nm, size, pos in parts:
            bg = bn.add_geom()
            bg.name, bg.type, bg.size, bg.pos = f"{bname}_{nm}", mujoco.mjtGeom.mjGEOM_BOX, size, pos
            bg.rgba, bg.group, bg.friction = rgba, 0, [1.0, 0.005, 0.0001]
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
                 target_visible=True, random_start=0.0, cmd_smooth=1.0, edge_bias=0.0, grasp_start=0.0, lift_reward="height",
                 grip_mode="delta", privileged=False, light_log=0.0, light_choices=None, eye_adapt=False,
                 light_gradient=0.0, light_tint=None, smell_off=False, smell_noise_x=1.0, **_):
        assert task in ("reach", "lift", "place", "sort", "smell"), task
        if task == "lift" and max_steps == 60:
            max_steps = 100                      # reach + descend + close + lift needs ~3 s
        if task in ("place", "sort", "smell") and max_steps == 60:
            max_steps = 150                      # + carry, lower and let go
        if task in ("sort", "smell"):
            max_distractors = 0                  # the other two bins are the clutter
        if not os.path.exists(PANDA_XML):
            raise FileNotFoundError(f"{PANDA_XML} not found. Run: python scripts/download_panda.py")
        self.task, self.max_steps, self.render_mode = task, max_steps, render_mode
        self.eye_noise, self.body_noise, self.light_jitter = eye_noise, body_noise, light_jitter
        self.max_dist, self.target_visible = max_distractors, target_visible
        self.n_decoys = 2
        bins = [("bin", [0.75, 0.12, 0.1, 1])] if task == "place" else \
            [(f"bin_{n}", [0.75, 0.12, 0.1, 1]) for n in ("left", "centre", "right")] if task == "sort" else \
            [(f"bin_{n}", [0.75, 0.12, 0.1, 1]) for n in ("ripe", "rotten")] if task == "smell" else []
        self.model = _build_model(self.n_decoys, n_facets, bins=bins)
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
        # finger collision pads (touch sense, grasp check)
        self.finger_geoms = [{g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f).id and m.geom_contype[g]}
                             for f in ("left_finger", "right_finger")]

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
        self.bin_mocaps = [m.body_mocapid[m.body(b).id] for b, _ in bins]
        specs = [BIN] * len(bins)
        for (b, _), sp in zip(bins, specs):
            for nm in ("floor", "n", "s", "e", "w"):
                self.spectrum[m.geom(f"{b}_{nm}").id] = sp
        if bins:
            self.bin_mocap = self.bin_mocaps[0]          # the TARGET bin (sort: the block's colour)
        self._colour = 0                                 # sort: index into SORT_COLOURS
        self._block_spec = TARGET
        self.n_priv = n_priv(task)

        self.n_vis = 2 * n_facets * N_CH
        # smell (task smell): per antenna and glomerulus type, the receptor response (left antenna first)
        self.n_smell = 0
        if task == "smell":
            t = np.load(SMELL_TABLE)
            self.smell_resp = t["response"]                        # (glomerulus, odorant)
            self.smell_recipes = t["recipes"]                      # (fruit, odorant)
            self.smell_drive = self.smell_recipes @ self.smell_resp.T   # (fruit, glomerulus) at full strength
            self.n_smell = 2 * self.smell_resp.shape[0]
        self.smell_off, self.smell_noise_x = smell_off, smell_noise_x
        self._fruit = 0
        self.n_brain_obs = 2 * self.n_vis + 13 + self.n_smell   # what the fly brain sees
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.n_brain_obs + (self.n_priv if privileged else 0),),
                                            np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.random_start = random_start
        self.edge_bias = edge_bias          # training only: share of blocks put near the table edges / corners
        # training only (lift): share of episodes that start partway through the task, half with the
        # gripper open around the block, half with the block already held (practice closing / lifting)
        self.grasp_start = grasp_start
        assert lift_reward in ("height", "goal"), lift_reward
        self.lift_reward = lift_reward
        # grip_mode: "delta" = each command nudges the finger target (old); "absolute" = the command
        # IS the finger target (-1 closed .. +1 open), as in ManiSkill / Fetch: a closed hand stays closed
        assert grip_mode in ("delta", "absolute"), grip_mode
        self.grip_mode = grip_mode
        # privileged: append the true state (block, goal, grasp ...) AFTER the brain's inputs, for the
        # training-only critic (asymmetric actor-critic). The fly brain never receives these numbers.
        self.privileged = privileged
        # lighting: by default brightness x (1 +- light_jitter). light_log > 0 (training): brightness x
        # exp(uniform(-light_log, light_log)), e.g. ln 2 = anything from half to double. light_choices
        # (tests): brightness is one of these factors, picked at random each episode.
        self.light_log, self.light_choices = light_log, light_choices
        # eye_adapt: photoreceptor light adaptation (robot property, training and evaluation). Each
        # photoreceptor type of each eye scales its gain to the average light it currently receives, so
        # the brain gets contrast rather than absolute brightness (fly photoreceptors adapt like this).
        # Photon noise is added BEFORE adaptation and grows like sqrt(light), so dim scenes are noisier.
        self.eye_adapt = eye_adapt
        self._adapt = None
        # tests only: uneven light, brighter towards one side of each eye's view and dimmer towards the
        # other (x exp(+-light_gradient) at the edges, side random per episode), and tinted light
        # (each receptor type gets its own factor, e.g. (0.7, 1.4, 1.0) = less broadband, more UV)
        self.light_gradient, self.light_tint = light_gradient, light_tint
        self._grad_sign = 1.0
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

    def eye_image(self, update=True):
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
            objs = [(self.block_geom, self._block_spec)] if self.target_visible else []
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
        if self.light_gradient:
            az = self.facet_az
            u = (az - az.mean()) / (np.ptp(az) / 2 + 1e-9)
            img = img * np.exp(self._grad_sign * self.light_gradient * u)[:, None]
        if self.light_tint is not None:
            img = img * np.asarray(self.light_tint)[None, :]
        if not self.eye_adapt:
            return np.clip(img * self._light, 0, 1.5)
        lit = img * self._light
        if update and self.eye_noise > 0:              # photon (shot) noise, before adaptation
            lit = np.maximum(lit + self.rng.normal(0, 1, lit.shape) * self.eye_noise * np.sqrt(np.maximum(lit, 0) / ADAPT_REF), 0)
        F = self.n_facets
        mean = np.stack([lit[:F].mean(0), lit[F:].mean(0)])             # (eye, receptor type)
        state = mean if self._adapt is None else self._adapt + ADAPT_RATE * (mean - self._adapt)
        if update:
            self._adapt = state
        gain = ADAPT_REF / (state + 1e-3)                                # per eye and receptor type
        out = np.concatenate([lit[:F] * gain[0], lit[F:] * gain[1]])
        return np.clip(out, 0, 1.5)

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

    def teacher_action(self):
        """The imitation teacher's move for the current situation (called across processes)."""
        return place_teacher(self) if self.task in ("place", "sort", "smell") else lift_teacher(self)

    def bin_pos(self):
        """Centre of the bin floor (on the table)."""
        return self.data.mocap_pos[self.bin_mocap].copy()

    def block_touched(self):
        """Any finger touching the block."""
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            if self.block_geom in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.block_geom else c.geom1
                if other in self.finger_geoms[0] or other in self.finger_geoms[1]:
                    return True
        return False

    def in_wrong_bin(self):
        """sort: block resting inside one of the OTHER bins."""
        b = self.block_pos()
        return any(np.all(np.abs(b[:2] - self.data.mocap_pos[mc][:2]) < BIN_IN) and b[2] < REST_Z + 0.012
                   for mc in self.bin_mocaps if mc != self.bin_mocap)

    def in_bin(self):
        """Block resting inside the bin (centre within BIN_IN in x and y, down on the floor)."""
        b, c = self.block_pos(), self.bin_pos()
        return bool(np.all(np.abs(b[:2] - c[:2]) < BIN_IN) and b[2] < REST_Z + 0.012)

    def finger_touch(self):
        """(left touching anything, right touching anything, both fingers on the block)."""
        d = self.data
        touch, on_block = [False, False], [False, False]
        for i in range(d.ncon):
            c = d.contact[i]
            for k, fg in enumerate(self.finger_geoms):
                if c.geom1 in fg or c.geom2 in fg:
                    other = c.geom2 if c.geom1 in fg else c.geom1
                    if other in fg or other in self.finger_geoms[1 - k]:
                        continue
                    touch[k] = True
                    if other == self.block_geom:
                        on_block[k] = True
        return touch[0], touch[1], on_block[0] and on_block[1]

    def is_grasped(self):
        return self.finger_touch()[2] and self.gripper_width() > 0.01

    def lift_height(self):
        return float(self.block_pos()[2] - self._block_z0)

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
        if self.eye_noise > 0 and not self.eye_adapt:     # (adapting eyes add photon noise inside eye_image)
            img = img + self.rng.normal(0, self.eye_noise, img.shape)
        img = img.ravel()
        d_img = img - (self._prev_img if self._prev_img is not None else img)
        self._prev_img = img
        p, v = self.tcp_pos(), self.tcp_vel()
        if self.body_noise > 0:
            p = p + self.rng.normal(0, self.body_noise, 3)
            v = v + self.rng.normal(0, 2 * self.body_noise, 3)
        body = np.concatenate([(p - 0.5 * (WS_LO + WS_HI)) * 4, v * 2, [self.gripper_width() * 25 - 1],
                               [float(self._grip_ctrl > 127)], [self._touch_obs()], self._last_action])
        out = [img, d_img, body]
        if self.n_smell:
            out.append(self.smell_obs())
        if self.privileged:
            out.append(self.privileged_state())
        return np.concatenate(out).astype(np.float32)

    def smell_obs(self):
        """Two antennae x glomerulus types: the fly's receptor responses to the fruit's smell here."""
        if self.smell_off:
            return np.zeros(self.n_smell)
        tcp, blk = self.tcp_pos(), self.block_pos()
        out = []
        for a in SMELL_ANTENNAE:
            d2 = float(np.sum((tcp + a - blk) ** 2))
            conc = np.exp(-d2 / (2 * SMELL_SIGMA ** 2)) * np.exp(self.rng.normal(0, SMELL_FLICKER)) + 0.02
            drive = conc * self.smell_drive[self._fruit]
            out.append(np.tanh(2.0 * drive) + self.rng.normal(0, SMELL_NOISE * self.smell_noise_x, drive.shape))
        return np.concatenate(out)

    def privileged_state(self):
        """True state for the training-only critic (never given to the fly brain)."""
        tcp, blk = self.tcp_pos(), self.block_pos()
        goal = getattr(self, "_goal", blk + np.array([0, 0, LIFT_GOAL]))
        l, r, _ = self.finger_touch()
        out = [tcp, blk, blk - tcp, goal - blk,
               [float(self.is_grasped()), self.lift_height() if hasattr(self, "_block_z0") else 0.0,
                self.gripper_width(), float(l), float(r)]]
        if self.task in ("place", "sort", "smell"):
            c = self.bin_pos()
            out += [c, blk - (c + np.array([0, 0, REST_Z]))]
        if self.task == "smell":
            out += [np.eye(2)[self._fruit]]
        if self.task == "sort":
            out += [np.eye(3)[self._colour]]
        return np.concatenate(out)

    def _touch_obs(self):
        """Touch: 0 none, 0.5 one finger, 1 both fingers press on something (reach: unused, 0)."""
        if self.task == "reach":
            return 0.0
        l, r, _ = self.finger_touch()
        return 0.5 * (l + r)

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
        if self.light_choices:
            self._light = float(self.rng.choice(self.light_choices))
        elif self.light_log > 0:
            self._light = float(np.exp(self.rng.uniform(-self.light_log, self.light_log)))
        else:
            self._light = 1 + self.rng.uniform(-self.light_jitter, self.light_jitter)
        self._tile = self.rng.uniform(0.06, 0.14)
        if self.light_gradient:                         # (drawn only then: other layouts stay as before)
            self._grad_sign = 1.0 if self.rng.random() < 0.5 else -1.0
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
        placed = [target]
        if self.task == "sort":                         # block colour, then three bins and the block
            self._colour = int(self.rng.integers(3))
            self._block_spec = SORT_COLOURS[self._colour][1]
            m.geom_rgba[self.block_geom] = SORT_COLOURS[self._colour][2]
            self.bin_mocap = self.bin_mocaps[self._colour]
            cs = [c + self.rng.uniform(-SORT_JITTER, SORT_JITTER, 2) for c in SORT_BINS]
            target = self.rng.uniform(SORT_BLOCK_LO, SORT_BLOCK_HI)
            for mc, c in zip(self.bin_mocaps, cs):
                d.mocap_pos[mc] = [c[0], c[1], 0.0]
            placed += cs
        if self.task == "smell":                        # ripe or rotten; two bins; the fruit on the near half
            self._fruit = int(self.rng.integers(2))
            self.bin_mocap = self.bin_mocaps[self._fruit]
            cs = [c + self.rng.uniform(-SORT_JITTER, SORT_JITTER, 2) for c in SMELL_BINS]
            target = self.rng.uniform(SORT_BLOCK_LO, SORT_BLOCK_HI)
            for mc, c in zip(self.bin_mocaps, cs):
                d.mocap_pos[mc] = [c[0], c[1], 0.0]
            placed += cs
        if self.task == "place":                        # the bin: somewhere else on the table
            for _ in range(200):
                binc = self.rng.uniform(BIN_LO, BIN_HI)
                if np.linalg.norm(binc - target) > BIN_MIN_GAP:
                    break
            d.mocap_pos[self.bin_mocap] = [binc[0], binc[1], 0.0]
            placed.append(binc)
        self._place(self.block_q, target)
        n_d = self.rng.integers(0, self.max_dist + 1) if self.max_dist > 0 else 0
        for i, qa in enumerate(self.decoy_q):
            if i < n_d:
                for _ in range(100):
                    xy = self.rng.uniform(BLOCK_LO, BLOCK_HI)
                    if min(np.linalg.norm(xy - p) for p in placed) > (0.13 if self.task == "place" else 0.1):
                        break
                placed.append(xy)
                self._place(qa, xy)
            else:
                d.qpos[qa:qa + 3] = [0, 0, -5]              # parked under the table, unseen
        # gripper starts somewhere above the table (or near the block: curriculum)
        start = self.rng.uniform([0.35, -0.2, 0.18], [0.65, 0.2, 0.35])
        if self.random_start > 0 and self.rng.random() < self.random_start:
            start = np.array([*target, 0.0]) + self.rng.uniform([-0.08, -0.08, 0.05], [0.08, 0.08, 0.12])
        mid = held = False
        if self.task in ("lift", "place", "sort", "smell") and self.grasp_start > 0 and self.rng.random() < self.grasp_start:
            mid, held = True, self.rng.random() < 0.5
            start = np.array([target[0], target[1], BLOCK_H]) + self.rng.uniform([-0.008, -0.008, -0.004],
                                                                                   [0.008, 0.008, 0.006])
        q = self._ik_solve(start, HOME, iters=60)
        d.qpos[:7] = q
        d.ctrl[:7] = q
        d.ctrl[7] = self._grip_ctrl = 255.0
        if mid:                                             # align the block with the fingers
            yaw = self.rng.uniform(-0.3, 0.3)
            d.qpos[self.block_q + 3:self.block_q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(m, d)
        for _ in range(50):                                 # let blocks settle
            mujoco.mj_step(m, d)
        if held:                                            # close the fingers on the block
            d.ctrl[7] = self._grip_ctrl = 0.0
            for _ in range(150):
                mujoco.mj_step(m, d)
        if n_d < len(self.decoy_q):                          # keep parked decoys parked
            for qa in self.decoy_q[n_d:]:
                d.qpos[qa:qa + 3] = [0, 0, -5]
                d.qvel[:] = 0
            mujoco.mj_forward(m, d)
        self._block_z0 = float(self.block_pos()[2])
        self._goal = self.block_pos() + np.array([0, 0, LIFT_GOAL])     # straight above the start
        if self.task in ("place", "sort", "smell"):                      # where the block should end up
            self._goal = self.bin_pos() + np.array([0, 0, REST_Z])
        self._target = self.tcp_pos()
        self._cmd = np.zeros(3)
        self._hold = 0
        self._last_action = np.zeros(4)
        self._steps = 0
        self._prev_img = None
        self._adapt = None                                  # eyes adapt to this scene from the first look
        return self._get_obs(), {}

    def _reward(self):
        if self.task == "lift":
            return self._reward_lift()
        if self.task in ("place", "sort", "smell"):
            return self._reward_place()
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

    def _reward_lift(self):
        d_t = float(np.linalg.norm(self.tcp_pos() - self.block_pos()))   # TCP to block centre
        held = self.is_grasped()
        lift = self.lift_height()
        r = 1 - np.tanh(5 * d_t)                                      # reach the block
        if self.lift_reward == "goal":
            g = float(np.linalg.norm(self.block_pos() - self._goal))
            if held:
                r += 1.0 + 2.0 * (1 - np.tanh(10 * g))                # grasp, then bring it up to the goal
            up = held and g < GOAL_DIST
        else:
            if held:
                r += 1.0 + 2.0 * float(np.clip(lift / LIFT_GOAL, 0, 1))   # grasp, then lift
            up = held and lift >= LIFT_GOAL
        self._hold = self._hold + 1 if up else 0
        success = self._hold >= LIFT_HOLD
        if success:
            r = 5.0
        return r / 5.0, success, d_t

    def _reward_place(self):
        """Staged (ManiSkill style), every stage pays more than the one before:
        reach the block (0..1) -> hold it (+1) -> carry it over the bin (+2 x closeness in x/y)
        -> lower it over the bin (+0.5) -> resting in the bin, still held (4.5) -> let go (5)
        -> stays there, released, 0.2 s: success (6). Divided by 6."""
        tcp, blk, c = self.tcp_pos(), self.block_pos(), self.bin_pos()
        d_t = float(np.linalg.norm(tcp - blk))
        held = self.is_grasped()
        dxy = float(np.linalg.norm(blk[:2] - c[:2]))
        r = 1 - np.tanh(5 * d_t)
        if held:
            r += 1.0 + 2.0 * (1 - np.tanh(5 * dxy))
            if dxy < 0.03:
                r += 0.5 * (1 - np.tanh(10 * max(0.0, blk[2] - REST_Z)))
        inside = self.in_bin()
        released = inside and not self.block_touched()
        if inside:
            r = 5.0 if released else 4.5
        self._hold = self._hold + 1 if released else 0
        success = self._hold >= LIFT_HOLD
        if success:
            r = 6.0
        return r / 6.0, success, d_t

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
        if self.grip_mode == "absolute":
            self._grip_ctrl = float((a[3] + 1) * 127.5)                        # -1 closed .. +1 open
        else:
            self._grip_ctrl = float(np.clip(self._grip_ctrl + a[3] * 64, 0, 255))   # 4 steps to open/close
        d.ctrl[7] = self._grip_ctrl
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(m, d)
        self._steps += 1
        self._last_action = a
        r, success, d_t = self._reward()
        truncated = self._steps >= self.max_steps
        info = {"is_success": success, "dist": d_t}
        if self.task in ("lift", "place", "sort", "smell"):
            info.update(grasped=self.is_grasped(), lift=self.lift_height(),
                        goal_dist=float(np.linalg.norm(self.block_pos() - self._goal)))
        if self.task in ("place", "sort", "smell"):
            info.update(in_bin=self.in_bin())
        if self.task == "smell":
            info.update(fruit=("ripe", "rotten")[self._fruit], wrong_bin=self.in_wrong_bin())
        if self.task == "sort":
            info.update(colour=SORT_COLOURS[self._colour][0], wrong_bin=self.in_wrong_bin())
        return self._get_obs(), float(r), False, truncated, info

    # ------------------------------------------------------------------ rendering
    def eye_view(self, size=190):
        img = self.eye_image(update=False)
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
    if env.task in ("place", "sort", "smell"):
        return place_teacher(env)
    if env.task == "lift":
        return _lift_controller(env)
    err = env.reach_point() - env.tcp_pos()
    a = np.zeros(4)
    a[:3] = np.clip(err / STEP * 0.5, -1, 1)
    a[3] = 1.0
    return a


def _lift_controller(env):
    """Scripted grasp: over the block (open) -> down to it -> close -> lift."""
    tcp, blk = env.tcp_pos(), env.block_pos()
    a = np.zeros(4)
    xy_err = np.linalg.norm((blk - tcp)[:2])
    closing = env._grip_ctrl < 200 or (xy_err < 0.01 and abs(tcp[2] - blk[2]) < 0.012)
    if not closing:
        goal = blk + np.array([0, 0, 0.0 if xy_err < 0.008 else 0.08])
        a[3] = 1.0
    else:
        a[3] = -1.0
        goal = blk.copy()
        if env._grip_ctrl <= 0 and env.finger_touch()[2]:
            goal = (env._goal.copy() if env.lift_reward == "goal"
                    else np.array([blk[0], blk[1], env._block_z0 + LIFT_GOAL + 0.03]))
            goal = goal - (blk - tcp)                  # move the hand, keep the hold offset
    a[:3] = np.clip((goal - tcp) / STEP * 0.5, -1, 1)
    return a


def lift_teacher(env):
    """Teacher for imitation (DAgger). Uses the TRUE block position (the fly brain never does)
    and decides only from the current situation, so it also gives the right move in situations
    the student got itself into: rise if low and off-target, line up above the block with the
    hand open, descend, close when level with the block, then bring the held block to the goal
    point straight above its start. If the block slips out, it starts again from that state."""
    tcp, blk = env.tcp_pos(), env.block_pos()
    a = np.zeros(4)
    if env.is_grasped():
        goal = getattr(env, "_goal", blk + np.array([0, 0, LIFT_GOAL]))
        target = goal - (blk - tcp)                       # move the hand so the block reaches the goal
        a[3] = -1.0
    else:
        xy_err = float(np.linalg.norm((blk - tcp)[:2]))
        dz = float(tcp[2] - blk[2])
        if xy_err > 0.012:
            if dz < 0.045:                                # low and off-target: go up first (no collisions)
                target = np.array([tcp[0], tcp[1], blk[2] + 0.08])
            else:                                         # line up above the block
                target = np.array([blk[0], blk[1], max(tcp[2], blk[2] + 0.06)])
            a[3] = 1.0
        elif dz > 0.008:                                  # straight down, hand open
            target = blk.copy()
            a[3] = 1.0
        else:                                             # level with the block: close
            target = blk.copy()
            a[3] = -1.0
    a[:3] = np.clip((target - tcp) / STEP * 0.6, -1, 1)
    return a


def place_teacher(env):
    """Pick-and-place teacher (TRUE block and bin positions; training targets only). Stateless:
    not holding -> grasp as in lift_teacher; holding -> lift to carry height, move over the bin,
    lower onto its floor, open; block in the bin and released -> move the hand up, open."""
    tcp, blk, c = env.tcp_pos(), env.block_pos(), env.bin_pos()
    a = np.zeros(4)
    if env.is_grasped():
        a[3] = -1.0
        dxy = float(np.linalg.norm(blk[:2] - c[:2]))
        carry = REST_Z + PLACE_CARRY
        if dxy > 0.012:
            if blk[2] < REST_Z + 0.05:                    # low: lift first (clear the rim)
                want = np.array([blk[0], blk[1], carry])
            else:                                         # carry over the bin
                want = np.array([c[0], c[1], max(blk[2], carry)])
        else:
            want = np.array([c[0], c[1], REST_Z])         # straight down onto the floor
            if blk[2] < REST_Z + 0.008:
                a[3] = 1.0                                # down: let go
        target = want - (blk - tcp)                       # move the hand, keep the hold offset
    elif env.in_bin():
        target = np.array([tcp[0], tcp[1], REST_Z + 0.15])   # done: hand up and away, open
        a[3] = 1.0
    else:
        return lift_teacher(env)
    a[:3] = np.clip((target - tcp) / STEP * 0.6, -1, 1)
    return a
