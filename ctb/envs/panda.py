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
    harvest greenhouse phase 1: one cherry tomato (3.2 cm) hangs from a tomato plant on its pedicel (stalk),
            plant somewhere along the far side; the hand works sideways, fingers pointing into the plant
            row, as greenhouse harvesting robots do. Grasp the tomato, pull it off the vine (the stalk snaps
            at its joint once the held tomato is pulled hard enough: tomatoes are picked by
            snapping this joint, SNAP_FORCE), carry it to the basket beside the robot and drop it in. Touching the plant
            is penalised (a broken stem costs a grower weeks of fruit). The tomato smells ripe (smell inputs,
            as in smell). Success = resting in the basket, released, 0.2 s (200 steps = 8 s).
    pickripe greenhouse phase 2: a truss of three cherry tomatoes side by side (6.5 cm apart), each green,
            turning or ripe (at least one ripe, at least one not); pick ONLY the ripe ones into the basket
            (one at a time), leave the others on the plant. Colours as fly photoreceptors would see them
            (ripe red is dark to a fly, turning is less green, green is like the leaves); riper tomatoes smell
            stronger. Picking an unripe tomato fails the episode. Success = every ripe tomato resting in the
            basket, released, 0.2 s, none unripe picked (300 steps = 12 s).

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
# wrist_cam="rgbd": the wrist eye is replaced by a close-range depth camera like the Intel RealSense D405 that
# harvesting robots carry on the hand (ideal range 7-50 cm, about +-2 % at 50 cm, global shutter). Its image,
# 16 x 10 pixels of (red, distance, green), goes into the SAME 480 wrist-eye input neurons of the fly brain:
# each pixel to the facet that points the same way, so the visual pathways keep their layout.
CAM_W, CAM_H, CAM_FOVY, CAM_OVERSAMPLE = 16, 10, 50.0, 4    # 50 x 73 deg view (D405: 58 x 87), 5 deg per pixel
                                                            # (the fly eye: ~11 deg per facet)
CAM_MIN, CAM_MAX = 0.07, 0.50                               # m: closer = no depth (hole), further = "far"
CAM_DEPTH_NOISE, CAM_HOLES = 0.01, 0.03                     # depth noise (x distance), share of missing pixels
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


# harvest (greenhouse phase 1): the plant is a static stem with one branch, a pedicel and three leaves
# behind the stem; the tomato is welded (softly) to the pedicel tip until it is picked
TOM_R = 0.016                # cherry tomato radius (3.2 cm across; the gripper opens to 8 cm)
TOM_MASS = 0.015             # kg
PED_LEN = 0.04               # pedicel: tomato top -> branch
BRANCH_LEN = 0.10            # branch: main stem -> pedicel (the stem stands this far behind the tomato)
TOM_LO = np.array([0.50, -0.14, 0.13])   # where the tomato hangs (centre)
TOM_HI = np.array([0.58, 0.14, 0.25])
SNAP_FORCE = 1.5             # N: held (closed hand) and pulled (or bent sideways) this hard, the stalk snaps at its
                             # joint; pushing the tomato up into the stalk does not snap it. Its own weight: 0.15 N
GRIP_X = 4.0                 # harvest: finger force x4 (the model's ~1.5 N squeeze is far below a real Panda's 70 N)
STALK_SOLREF = 0.02          # stalk springiness (constraint time constant, s)
BASKET = np.array([0.44, -0.25])          # basket on the robot's cart, beside the base (+- SORT_JITTER)
BASKET_RIM = 0.04            # deeper than the bins: a round tomato must not roll out
HARVEST_CARRY = 0.09         # teacher: TCP height when dropping the tomato into the basket
PRE_GRASP = 0.07             # teacher: line up this far back along the approach line before going in
# harvest hand: tilted 30 deg from straight down towards the plant (fingers point forward-down, open along y),
# the way harvesting arms come at hanging fruit from the front and above. Straight forward (horizontal) is
# out of the Panda's comfortable reach this low and close; 30 deg covers the whole fruit zone.
HARVEST_PITCH = np.radians(60)                    # finger direction, below horizontal
APPROACH = np.array([np.cos(HARVEST_PITCH), 0, -np.sin(HARVEST_PITCH)])
YAW_W = 0.3                  # IK weight of turning about the finger axis (vs tilting it): more freedom to stay off limits
HOME_HARVEST = np.array([0.0, -0.07, 0.0, -2.705, 0.0, 3.159, 0.785])   # arm posture with that hand
HOME_HARVEST_TCP = np.array([0.50, 0.0, 0.22])    # where it puts the TCP
PLANT_PENALTY = 0.5          # reward lost per step while the robot touches the plant
# pickripe (phase 2): three tomatoes on a truss that runs sideways (along y) at the end of the branch
TRUSS_Y = (0.0, -0.065, 0.065)            # tomato slots (slot 0 = the middle one, called "block")
TOM_LO3 = np.array([0.50, -0.08, 0.13])   # where the middle tomato hangs
TOM_HI3 = np.array([0.58, 0.08, 0.25])
RIPENESS = ("green", "turning", "ripe")
RIPE_RGBA = ([0.36, 0.62, 0.18, 1], [0.92, 0.50, 0.14, 1], [0.85, 0.12, 0.08, 1])
RIPE_SMELL = (0.1, 0.4, 1.0)              # how strongly each stage smells of ripe fruit (fewer esters when unripe)
PICK_OPEN = 0.35                          # teacher: open the hand only this far (5.5 cm) between neighbours
LOOK_BACK = 0.12                          # teacher: first come to 12 cm in front of the middle of the truss
LOOK_NEAR = 0.13                          # (from further away the fly eye cannot tell the three tomatoes apart)
BASKET3 = np.array([0.48, -0.28])         # phase 2: the basket stands on a post on the cart, 10 cm up (a long
BASKET3_Z = 0.10                          # carry at table height twisted the arm into its joint limits)


def n_priv(task):
    """Size of privileged_state() for a task (place adds the bin position and block-to-bin;
    sort also the block's colour, one-hot)."""
    return N_PRIV + {"place": 6, "sort": 9, "smell": 8, "harvest": 8, "pickripe": 18}.get(task, 0)


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
# harvest. Approximate photoreceptor views. Flies see little beyond ~600 nm, so a ripe red tomato is
# dark to them and stands out from the plant mainly by its lack of green; leaves reflect green, little UV.
PLANT = np.array([0.30, 0.10, 0.55])
TOMATO_RED = np.array([0.22, 0.04, 0.10])
TOMATO_SPECS = (np.array([0.36, 0.10, 0.50]),   # green: like the leaves, a little brighter (glossy skin)
                np.array([0.30, 0.05, 0.28]),   # turning (orange-pink): less green
                TOMATO_RED)
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


def _vis_geom(body, gtype, rgba, size=None, pos=None, quat=None, fromto=None, material=None):
    """A geom for people watching only: drawn (group 2), never touched, never seen by the fly eyes."""
    g = body.add_geom()
    g.type, g.rgba, g.group, g.contype, g.conaffinity, g.density = gtype, rgba, 2, 0, 0, 0
    if fromto is not None:
        g.fromto = fromto
        g.size = [size[0], 0, 0]
    else:
        g.size = list(size) + [0] * (3 - len(size))
        g.pos = pos
        if quat is not None:
            g.quat = quat
    if material:
        g.material = material
    return g


def _axes_quat(x, z_hint=(0, 0, 1.0)):
    """Quaternion of a frame whose x axis is x and whose z axis is as close to z_hint as possible."""
    x = np.asarray(x, float) / np.linalg.norm(x)
    y = np.cross(np.asarray(z_hint, float), x)
    y = y / (np.linalg.norm(y) + 1e-9)
    z = np.cross(x, y)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([x, y, z], 1).ravel())
    return q


def _tomato_plant_look(body, seed, y0=0.0, ripe_truss=False, back_only=False):
    """A greenhouse tomato plant for the videos (visual only), stem base at (0, y0, 0) of body: a slightly
    crooked main stem on a support string, compound leaves (rachis + leaflet pairs + end leaflet) from 14 cm
    up (lower leaves removed, as growers do), a cluster of green fruit, small young fruit, yellow flowers
    and the stub of a harvested truss."""
    rng = np.random.default_rng(seed)
    STEM, LEAF, STRING = [0.30, 0.47, 0.17, 1], [0.17, 0.40, 0.12, 1], [0.85, 0.80, 0.62, 1]
    pts = [np.array([0, y0, 0.06])]
    for z in np.arange(0.14, 0.86, 0.08):
        pts.append(np.array([rng.uniform(-0.004, 0.004), y0 + rng.uniform(-0.007, 0.007), z]))
    for k in range(len(pts) - 1):                         # main stem, thinner towards the top
        r = 0.010 - 0.005 * k / (len(pts) - 1)
        _vis_geom(body, mujoco.mjtGeom.mjGEOM_CAPSULE, STEM, size=[r], fromto=[*pts[k], *pts[k + 1]], material="plant")
    _vis_geom(body, mujoco.mjtGeom.mjGEOM_CAPSULE, STRING, size=[0.0012], fromto=[*pts[-1], 0, y0, 1.9])

    def stem_at(z):
        k = min(int((z - 0.06) / 0.08), len(pts) - 2)
        f = np.clip((z - pts[k][2]) / (pts[k + 1][2] - pts[k][2]), 0, 1)
        return pts[k] + f * (pts[k + 1] - pts[k])

    lo, hi = (-80, 80) if back_only else (-115, 115)
    for k, z in enumerate([0.14, 0.22, 0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.78]):
        a = np.radians(((35 + 137.5 * k + 180) % 360) - 180)
        a = np.radians(np.clip(np.degrees(a), lo, hi))
        L = (0.24 if z < 0.6 else 0.17) * rng.uniform(0.85, 1.1)
        d = np.array([np.cos(a), np.sin(a), 0])
        p0 = stem_at(z)
        p1 = p0 + 0.5 * L * (d + np.array([0, 0, 0.45]))
        p2 = p1 + 0.5 * L * (d + np.array([0, 0, -0.55]))
        for u, v in ((p0, p1), (p1, p2)):                    # rachis
            _vis_geom(body, mujoco.mjtGeom.mjGEOM_CAPSULE, STEM, size=[0.0024], fromto=[*u, *v], material="plant")
        col = [c * rng.uniform(0.9, 1.1) for c in LEAF[:3]] + [1]
        for f, sc in ((0.22, 1.0), (0.42, 1.3), (0.62, 1.4), (0.82, 1.2)):   # leaflet pairs
            seg = (p0, p1) if f < 0.5 else (p1, p2)
            g = f / 0.5 if f < 0.5 else (f - 0.5) / 0.5
            c = seg[0] + g * (seg[1] - seg[0])
            t = (seg[1] - seg[0]) / np.linalg.norm(seg[1] - seg[0])
            side = np.cross(t, [0, 0, 1.0])
            side /= np.linalg.norm(side) + 1e-9
            for sgn in (1, -1):
                ax = sgn * side + 0.6 * t
                _vis_geom(body, mujoco.mjtGeom.mjGEOM_ELLIPSOID, col, size=[0.030 * sc, 0.016 * sc, 0.0015],
                          pos=c + sgn * side * 0.026 * sc + t * 0.012 + [0, 0, -0.004],
                          quat=_axes_quat(ax, [0, 0, 1] + 0.3 * sgn * side), material="plant")
                _vis_geom(body, mujoco.mjtGeom.mjGEOM_ELLIPSOID, col, size=[0.013, 0.008, 0.0012],   # small leaflets
                          pos=c - t * 0.025 + sgn * side * 0.014, quat=_axes_quat(sgn * side + 0.3 * t), material="plant")
        t = (p2 - p1) / np.linalg.norm(p2 - p1)
        _vis_geom(body, mujoco.mjtGeom.mjGEOM_ELLIPSOID, col, size=[0.042, 0.022, 0.0015], pos=p2 + t * 0.034,
                  quat=_axes_quat(t), material="plant")

    def truss(z, az, n, r, rgba, flowers=False):
        d = np.array([np.cos(np.radians(az)), np.sin(np.radians(az)), 0])
        p0 = stem_at(z)
        p1 = p0 + d * 0.05 + [0, 0, 0.015]
        p2 = p1 + d * 0.06 + [0, 0, -0.03]
        for u, v in ((p0, p1), (p1, p2)):
            _vis_geom(body, mujoco.mjtGeom.mjGEOM_CAPSULE, STEM, size=[0.0022], fromto=[*u, *v], material="plant")
        for i in range(n):
            q = p1 + (p2 - p1) * (i + 0.5) / n
            off = np.cross(d, [0, 0, 1]) * (0.012 if i % 2 else -0.012)
            c = q + off + [0, 0, -0.012 - r]
            _vis_geom(body, mujoco.mjtGeom.mjGEOM_CAPSULE, STEM, size=[0.0012], fromto=[*q, *(c + [0, 0, r])], material="plant")
            if flowers:
                for j in range(5):
                    ang = 2 * np.pi * j / 5
                    pv = np.array([np.cos(ang), np.sin(ang), -0.6])
                    _vis_geom(body, mujoco.mjtGeom.mjGEOM_ELLIPSOID, [0.98, 0.85, 0.15, 1], size=[0.008, 0.003, 0.001],
                              pos=c + pv * 0.007, quat=_axes_quat(pv))
            else:
                _vis_geom(body, mujoco.mjtGeom.mjGEOM_SPHERE, rgba, size=[r * rng.uniform(0.85, 1.1)], pos=c,
                          material="fruit")

    g_az, f_az = (60, -70) if not back_only else (40, -40)
    truss(0.44, g_az, 4, 0.015, [0.42, 0.62, 0.18, 1])                       # green fruit
    if ripe_truss:
        truss(0.34, -g_az, 3, 0.016, [0.78, 0.30, 0.08, 1])                  # turning (orange-red)
    truss(0.60, f_az, 4, 0.007, [0.45, 0.68, 0.22, 1])                       # young fruit
    truss(0.74, 100 if not back_only else 30, 4, 0.0, None, flowers=True)    # flowers
    p0 = stem_at(0.10)                                                        # harvested truss: a bare stub
    _vis_geom(body, mujoco.mjtGeom.mjGEOM_CAPSULE, [0.45, 0.42, 0.22, 1], size=[0.0025],
              fromto=[*p0, *(p0 + [0.03, -0.02, -0.005])])


def _build_model(n_decoys, n_facets, bins=(), harvest=False, n_tom=1):
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

    blk = block("block", [0.95, 0.85, 0.1, 1])
    if harvest:                                  # the block is a cherry tomato on a plant
        tg = blk.geoms[0]
        tg.type, tg.size, tg.rgba, tg.mass = mujoco.mjtGeom.mjGEOM_SPHERE, [TOM_R, 0, 0], [0.85, 0.12, 0.08, 1], TOM_MASS
        tg.condim, tg.friction = 6, [1.2, 0.01, 0.002]    # skin: grips, and does not roll away forever
        tg.solref, tg.priority = [0.005, 1.0], 1          # firm contact (a light fruit otherwise creeps out)
        pl = wb.add_body()                       # origin: where the branch leaves the main stem
        pl.name, pl.mocap, pl.pos = "plant", True, [0.65, 0, 0.2]
        green, leaf = [0.30, 0.47, 0.17, 1], [0.17, 0.40, 0.12, 1]
        parts = [("stem", mujoco.mjtGeom.mjGEOM_CYLINDER, [0.007, 0.33, 0], [0, 0, -0.17], [1, 0, 0, 0], green),
                 ("branch", mujoco.mjtGeom.mjGEOM_CAPSULE, [0.003, BRANCH_LEN / 2, 0], [-BRANCH_LEN / 2, 0, 0],
                  [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0], green),
                 ("pedicel", mujoco.mjtGeom.mjGEOM_CAPSULE, [0.0018, PED_LEN / 2, 0], [-BRANCH_LEN, 0, -PED_LEN / 2],
                  [1, 0, 0, 0], green)]
        if n_tom > 1:                            # the truss: sideways at the end of the branch, one pedicel per tomato
            ty = max(abs(v) for v in TRUSS_Y[:n_tom])
            parts.append(("truss", mujoco.mjtGeom.mjGEOM_CAPSULE, [0.0025, ty, 0], [-BRANCH_LEN, 0, 0],
                          [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0], green))
            for k in range(1, n_tom):
                parts.append((f"pedicel{k}", mujoco.mjtGeom.mjGEOM_CAPSULE, [0.0018, PED_LEN / 2, 0],
                              [-BRANCH_LEN, TRUSS_Y[k], -PED_LEN / 2], [1, 0, 0, 0], green))
        for k, (lx, ly, lz) in enumerate([(0.04, 0.05, 0.07), (0.05, -0.05, -0.02), (0.04, 0.04, -0.11)]):
            parts.append((f"leaf{k}", mujoco.mjtGeom.mjGEOM_ELLIPSOID, [0.05, 0.03, 0.003], [lx, ly, lz],
                          _turn(np.array([0, 0, 1.0]), np.array([-0.5, 0, 1.0]) / np.sqrt(1.25)), leaf))
        for nm, typ, size, pos, quat, rgba in parts:
            g = pl.add_geom()
            g.name, g.type, g.size, g.pos, g.quat, g.rgba = f"plant_{nm}", typ, size, pos, quat, rgba
            g.group = 0
            if n_tom > 1 and nm.startswith(("pedicel", "truss")):
                g.contype = g.conaffinity = 0    # (phase 2) thin stalks give way to the hand: seen, not solid
        # what people see in the videos (visual only: the brain's eyes and the physics ignore it): a real-looking
        # plant around the working parts above, its neighbours in the row, the grow slab and white ground cover
        for nm, rgba, spec_ in (("plant", [0.3, 0.5, 0.2, 1], 0.15), ("fruit", [0.8, 0.1, 0.05, 1], 0.7)):
            mt = spec.add_material()
            mt.name, mt.specular, mt.shininess, mt.rgba = nm, spec_, 0.6 if nm == "fruit" else 0.1, rgba
        look = wb.add_body()
        look.name, look.mocap, look.pos = "plant_look", True, [0.65, 0, 0]
        _tomato_plant_look(look, 1, ripe_truss=n_tom == 1)
        _tomato_plant_look(look, 2, y0=0.42, back_only=True)
        _tomato_plant_look(look, 3, y0=-0.42, back_only=True)
        _vis_geom(look, mujoco.mjtGeom.mjGEOM_BOX, [0.93, 0.93, 0.9, 1], size=[0.07, 0.75, 0.03], pos=[0.02, 0, 0.03])
        _vis_geom(look, mujoco.mjtGeom.mjGEOM_BOX, [0.86, 0.87, 0.84, 1], size=[0.22, 1.0, 0.0012], pos=[0.11, 0, 0.0012])
        toms = [blk]
        for k in range(1, n_tom):                # more tomatoes on the truss (bodies tomato1, tomato2, ...)
            t = block(f"tomato{k}", [0.85, 0.12, 0.08, 1])
            g = t.geoms[0]
            g.type, g.size, g.mass = mujoco.mjtGeom.mjGEOM_SPHERE, [TOM_R, 0, 0], TOM_MASS
            g.condim, g.friction, g.solref, g.priority = 6, [1.2, 0.01, 0.002], [0.005, 1.0], 1
            toms.append(t)
        for k, t in enumerate(toms):
            t.geoms[0].material = "fruit"
            for j in range(5):                   # the tomato's calyx (green star of sepals on top)
                ang = 2 * np.pi * j / 5 + 0.3
                v = np.array([np.cos(ang), np.sin(ang), -0.35])
                _vis_geom(t, mujoco.mjtGeom.mjGEOM_ELLIPSOID, [0.22, 0.45, 0.12, 1], size=[0.009, 0.0035, 0.001],
                          pos=[0, 0, TOM_R * 0.93] + v * 0.006, quat=_axes_quat(v))
            eq = spec.add_equality()             # the stalk: the tomato's top hangs at the pedicel tip until picked
            eq.name = "stalk" if k == 0 else f"stalk{k}"
            eq.type, eq.objtype = mujoco.mjtEq.mjEQ_CONNECT, mujoco.mjtObj.mjOBJ_BODY
            eq.name1, eq.name2 = t.name, "plant"
            eq.data[:6] = [0, 0, TOM_R, -BRANCH_LEN, TRUSS_Y[k], -PED_LEN]
            eq.solref = [STALK_SOLREF, 1.0]      # a bit springy: a pull stretches it measurably
            ex = spec.add_exclude()
            ex.bodyname1, ex.bodyname2 = t.name, "plant"
    for k, bdef in enumerate(bins):              # shallow boxes the eyes see by ray (group 0), moved per episode
        bname, rgba = bdef[:2]
        bn = wb.add_body()
        bn.name, bn.mocap, bn.pos = bname, True, [0.5, 0.2 - 0.2 * k, 0]
        h, w, f, rim = BIN_HALF, BIN_WALL, BIN_FLOOR, (bdef[2] if len(bdef) > 2 else BIN_RIM_H)
        parts = [("floor", [h, h, f / 2], [0, 0, f / 2])]
        zc, hz = f + rim / 2, rim / 2
        parts += [("n", [h, w / 2, hz], [0, h - w / 2, zc]), ("s", [h, w / 2, hz], [0, -h + w / 2, zc]),
                  ("e", [w / 2, h - w, hz], [h - w / 2, 0, zc]), ("w", [w / 2, h - w, hz], [-h + w / 2, 0, zc])]
        for nm, size, pos in parts:
            bg = bn.add_geom()
            bg.name, bg.type, bg.size, bg.pos = f"{bname}_{nm}", mujoco.mjtGeom.mjGEOM_BOX, size, pos
            bg.rgba, bg.group, bg.friction = rgba, 0, [1.0, 0.005, 0.0001]
        if bname == "basket" and n_tom > 1:      # (phase 2) the post it stands on, drawn only
            _vis_geom(bn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.35, 0.35, 0.38, 1], size=[0.012, BASKET3_Z / 2],
                      pos=[0, 0, -BASKET3_Z / 2])
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
    wc = spec.body("hand").add_camera()          # (used only with wrist_cam="rgbd") beside the fingers, aimed at a
    look = np.array([0, 0, 0.13]) - WRIST_EYE_POS    # point 3 cm beyond the fingertips: sees between the fingers too
    zc = -look / np.linalg.norm(look)
    xc = np.array([0, 1.0, 0])
    qc = np.zeros(4)
    mujoco.mju_mat2Quat(qc, np.stack([xc, np.cross(zc, xc), zc], 1).ravel())
    wc.name, wc.pos, wc.quat, wc.fovy = "wrist_cam", WRIST_EYE_POS, qc, CAM_FOVY
    ws.quat = _turn(c_left, np.array([0, 0, 1.0]))                     # field centred along the fingers
    hg = head.add_geom()
    hg.type, hg.size, hg.rgba = mujoco.mjtGeom.mjGEOM_SPHERE, [0.04, 0, 0], [0.6, 0.15, 0.1, 1]
    if harvest:                                  # (dark grey here: a red ball would look like a tomato)
        hg.rgba = [0.18, 0.18, 0.22, 1]
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
    if harvest:                                  # videos: from the aisle beside the robot, facing the plant row
        hc = wb.add_camera()
        hc.name, hc.pos, hc.fovy = "aisle", [-0.15, -1.30, 0.90], 46
        if n_tom > 1:                            # (closer: three small tomatoes)
            hc.pos, hc.fovy = [0.0, -1.05, 0.72], 40
        hc.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
        hc.targetbody = "ws_centre"
    site = spec.body("hand").add_site()
    site.name, site.pos = "tcp", [0, 0, 0.1034]
    return spec.compile()


class FlyPandaEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, task="reach", n_facets=160, max_steps=60, seed=None, render_mode=None,
                 eye_noise=0.01, body_noise=0.005, light_jitter=0.2, max_distractors=2,
                 target_visible=True, random_start=0.0, cmd_smooth=1.0, edge_bias=0.0, grasp_start=0.0, return_start=0.0, lift_reward="height",
                 grip_mode="delta", privileged=False, light_log=0.0, light_choices=None, eye_adapt=False,
                 light_gradient=0.0, light_tint=None, smell_off=False, smell_noise_x=1.0, tomato_look=None,
                 wrist_cam=None, ripe_max=2, **_):
        assert task in ("reach", "lift", "place", "sort", "smell", "harvest", "pickripe"), task
        if task == "lift" and max_steps == 60:
            max_steps = 100                      # reach + descend + close + lift needs ~3 s
        if task in ("place", "sort", "smell") and max_steps == 60:
            max_steps = 150                      # + carry, lower and let go
        if task == "harvest" and max_steps == 60:
            max_steps = 200                      # line up, go in, close, pull, carry, drop
        if task == "pickripe" and max_steps == 60:
            max_steps = 300                      # up to two tomatoes
        if task in ("sort", "smell", "harvest", "pickripe"):
            max_distractors = 0                  # the other two bins are the clutter
        if not os.path.exists(PANDA_XML):
            raise FileNotFoundError(f"{PANDA_XML} not found. Run: python scripts/download_panda.py")
        self.task, self.max_steps, self.render_mode = task, max_steps, render_mode
        self.eye_noise, self.body_noise, self.light_jitter = eye_noise, body_noise, light_jitter
        self.max_dist, self.target_visible = max_distractors, target_visible
        self.n_decoys = 2
        bins = [("bin", [0.75, 0.12, 0.1, 1])] if task == "place" else \
            [(f"bin_{n}", [0.75, 0.12, 0.1, 1]) for n in ("left", "centre", "right")] if task == "sort" else \
            [(f"bin_{n}", [0.75, 0.12, 0.1, 1]) for n in ("ripe", "rotten")] if task == "smell" else \
            [("basket", [0.55, 0.35, 0.15, 1], BASKET_RIM)] if task in ("harvest", "pickripe") else []
        self.harvest = task in ("harvest", "pickripe")
        self.n_tom = 3 if task == "pickripe" else 1
        self.model = _build_model(self.n_decoys, n_facets, bins=bins, harvest=self.harvest, n_tom=self.n_tom)
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
        self.bin_mocaps = [m.body_mocapid[m.body(b[0]).id] for b in bins]
        specs = [BIN] * len(bins)
        for (b, *_), sp in zip(bins, specs):
            for nm in ("floor", "n", "s", "e", "w"):
                self.spectrum[m.geom(f"{b}_{nm}").id] = sp
        if bins:
            self.bin_mocap = self.bin_mocaps[0]          # the TARGET bin (sort: the block's colour)
        self._colour = 0                                 # sort: index into SORT_COLOURS
        self._block_spec = TARGET
        # the held object: a 5 cm block, or (harvest) a cherry tomato; the hand points down, or (harvest)
        # tilted towards the plant row (APPROACH), fingers opening along y
        self._obj_r = BLOCK_H * 1.2                      # seen as a ball of this radius
        self.rest_z = REST_Z                             # object centre height resting in a bin
        self._R_des = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])
        self._home = HOME
        self._null_gain = 0.05                           # pull towards the home posture (redundant 7th joint)
        self.att, self._tgt = [False], 0                 # harvest: tomatoes still on the vine; the current one
        self.ripe = [2]                                  # harvest: ripeness stage of each tomato (2 = ripe)
        self.tom_q, self.tom_geoms, self.stalks = [self.block_q], [self.block_geom], [0]
        self._unripe_picked = 0
        self._plant_hits = 0
        if self.harvest:
            self._block_spec, self._obj_r, self.rest_z = TOMATO_RED, TOM_R, BIN_FLOOR + TOM_R
            y = np.array([0, -1.0, 0])
            self._R_des = np.stack([np.cross(y, APPROACH), y, APPROACH], 1)
            self._home = HOME_HARVEST
            self._null_gain = 0.05                       # (the tilted hand lets joints 1 and 3 drift apart)
            self.plant_mocap = m.body_mocapid[m.body("plant").id]
            self.look_mocap = m.body_mocapid[m.body("plant_look").id]
            self.plant_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("plant").id}
            for g in self.plant_geoms:
                self.spectrum[g] = PLANT
            m.opt.cone, m.opt.impratio = mujoco.mjtCone.mjCONE_ELLIPTIC, 10.0   # firm friction: a round fruit
            m.actuator_gainprm[7, 0] *= GRIP_X              # stronger squeeze (position servo: gain and stiffness)
            m.actuator_biasprm[7, 1:3] *= GRIP_X
            names = ["block"] + [f"tomato{k}" for k in range(1, self.n_tom)]
            self.tom_q = [m.jnt_qposadr[m.body(b).jntadr[0]] for b in names]
            self.tom_geoms = [m.geom(b).id for b in names]
            self.stalks = [m.equality("stalk" if k == 0 else f"stalk{k}").id for k in range(self.n_tom)]
            for k, e in enumerate(self.stalks):                          # (the compiler sets this from the start pose)
                m.eq_data[e][3:6] = [-BRANCH_LEN, TRUSS_Y[k], -PED_LEN]
            self.att, self.ripe = [False] * self.n_tom, [2] * self.n_tom
            self.stalk = self.stalks[0]
            self.robot_bodies = set(range(1, m.body("hand").id + 1)) | {m.body(f).id for f in ("left_finger", "right_finger")}
        # (phase 2) a 2-5 mm stalk or stem between eye and tomato does not hide a 3.2 cm tomato (one test ray
        # through its centre otherwise hid it completely: from above, the stalk always crosses that ray)
        self._thin = {m.geom(f"plant_{n}").id for n in ["stem", "branch", "pedicel", "truss"] +
                      [f"pedicel{k}" for k in range(1, self.n_tom)]} if self.n_tom > 1 else set()
        self.n_priv = n_priv(task)

        self.n_vis = 2 * n_facets * N_CH
        # smell (task smell): per antenna and glomerulus type, the receptor response (left antenna first)
        self.n_smell = 0
        if task in ("smell", "harvest", "pickripe"):
            t = np.load(SMELL_TABLE)
            self.smell_resp = t["response"]                        # (glomerulus, odorant)
            self.smell_recipes = t["recipes"]                      # (fruit, odorant)
            self.smell_drive = self.smell_recipes @ self.smell_resp.T   # (fruit, glomerulus) at full strength
            self.n_smell = 2 * self.smell_resp.shape[0]
            glo = list(t["glomeruli"])                              # for the video panel only
            self._smell_groups = ([glo.index(g) for g in ("DM2", "DM3", "VM5v", "VM2") if g in glo],
                                  [glo.index(g) for g in ("DC4", "DP1l", "DP1m") if g in glo])
        self.smell_off, self.smell_noise_x = smell_off, smell_noise_x
        # tests only (pickripe): every tomato looks ripe to the eyes ("all_ripe"), so only smell can tell
        self.tomato_look = tomato_look
        assert wrist_cam in (None, "rgbd"), wrist_cam
        self.wrist_cam = wrist_cam
        self.ripe_max = ripe_max                # pickripe: at most this many ripe tomatoes (1 = choose one of three)
        self._cam_r, self._cam_gain, self._cam_last = None, None, None
        if wrist_cam:                           # which wrist-eye facet each camera pixel feeds (one to one, by layout)
            from scipy.optimize import linear_sum_assignment
            F = self.n_facets
            assert F == CAM_W * CAM_H, "wrist_cam needs 160 facets per eye (16 x 10 pixels)"
            az, el = self.facet_az[F:], self.facet_el[F:]
            fa = np.stack([(az - az.mean()) / np.ptp(az), (el - el.mean()) / np.ptp(el)], 1)
            cc, rr = np.meshgrid(np.arange(CAM_W), np.arange(CAM_H))
            px = np.stack([(cc.ravel() / (CAM_W - 1) - 0.5), (0.5 - rr.ravel() / (CAM_H - 1))], 1)
            _, self._cam_pix = linear_sum_assignment(((fa[:, None, :] - px[None, :, :]) ** 2).sum(-1))
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
        # training only (pickripe, two ripe): share of episodes that start after the first pick, one ripe tomato
        # already in the basket and the hand over it (practise the way back to the plant and the second pick)
        self.return_start = return_start
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
        self._cam = "aisle" if self.harvest else "overview"
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
            if e == 1 and self.wrist_cam:            # (filled in by the camera below)
                continue
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
            if self.harvest:                         # every tomato, coloured by its ripeness
                looks = [2] * self.n_tom if self.tomato_look == "all_ripe" else self.ripe
                objs = [(g, TOMATO_SPECS[r]) for g, r in zip(self.tom_geoms, looks)] if self.target_visible else []
            else:
                objs = [(self.block_geom, self._block_spec)] if self.target_visible else []
            objs += [(g, DECOY) for g in self.decoy_geoms if d.geom_xpos[g][2] > -1]
            r_eq = self._obj_r                        # a cube seen as a ball of about the same size
            gid1 = np.zeros(1, np.int32)
            sl = slice(e * F, (e + 1) * F)
            fdir = world[:F]
            for g, spec in sorted(objs, key=lambda o: -np.linalg.norm(d.geom_xpos[o[0]] - origin)):
                r_eq = self._obj_r if (g == self.block_geom or g in self.tom_geoms) else BLOCK_H * 1.2
                rel = d.geom_xpos[g] - origin
                dist_b = np.linalg.norm(rel)
                hit_d = mujoco.mj_ray(m, d, origin, rel / dist_b, self._occl_groups, 1, skip, gid1)
                if gid1[0] != g and 0 <= hit_d < dist_b - 0.03 and gid1[0] not in self._thin:   # hidden behind
                    continue                                                                # the arm, a leaf
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
            out = np.clip(img * self._light, 0, 1.5)
            if self.wrist_cam:
                out[self.n_facets:] = self.camera_image(update)
            return out
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
        out = np.clip(out, 0, 1.5)
        if self.wrist_cam:
            out[F:] = self.camera_image(update)
        return out

    def camera_image(self, update=True):
        """wrist_cam: (160, 3) wrist-eye inputs from the depth camera: per facet (red, nearness, green) of its pixel.
        Colour: the rendered image (scene light x brightness / tint / uneven light of the episode), auto exposure
        (overall gain towards a mean of ADAPT_REF, colour ratios kept), noise. Distance: metric depth with noise and
        missing pixels, nearness = 1 at 7 cm .. 0 at 50 cm, 0 where missing or nearer than 7 cm."""
        if not update and self._cam_last is not None:
            return self._cam_last
        if self._cam_r is None:
            self._cam_r = mujoco.Renderer(self.model, CAM_H * CAM_OVERSAMPLE, CAM_W * CAM_OVERSAMPLE)
            self._cam_opt = mujoco.MjvOption()
            self._cam_opt.geomgroup[:] = [1, 1, 0, 1, 1, 0]     # what the fly eye's rays see: plant, robot shapes, fruit
        r, k = self._cam_r, CAM_OVERSAMPLE
        saved = None
        if self.harvest and (self.tomato_look == "all_ripe" or not self.target_visible):   # tests: the camera too
            saved = self.model.geom_rgba[self.tom_geoms].copy()
            self.model.geom_rgba[self.tom_geoms] = RIPE_RGBA[2] if self.target_visible else [0, 0, 0, 0]
        r.update_scene(self.data, camera="wrist_cam", scene_option=self._cam_opt)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        rgb = r.render().astype(np.float64) / 255.0
        r.enable_depth_rendering()
        r.update_scene(self.data, camera="wrist_cam", scene_option=self._cam_opt)
        z = r.render().astype(np.float64)
        r.disable_depth_rendering()
        if saved is not None:
            self.model.geom_rgba[self.tom_geoms] = saved
        rgb = rgb.reshape(CAM_H, k, CAM_W, k, 3).mean((1, 3))           # pixel binning
        z = z.reshape(CAM_H, k, CAM_W, k).mean((1, 3))
        light = self._light * (np.ones(3) if self.light_tint is None else np.asarray(self.light_tint))
        rgb = rgb * light
        if self.light_gradient:
            u = np.linspace(-1, 1, CAM_W)[None, :, None]
            rgb = rgb * np.exp(self._grad_sign * self.light_gradient * u)
        lum = float(rgb.mean()) + 1e-3
        gain = ADAPT_REF[0] / lum if self._cam_gain is None else self._cam_gain + ADAPT_RATE * (ADAPT_REF[0] / lum - self._cam_gain)
        self._cam_gain = gain
        rgb = rgb * gain + self.rng.normal(0, self.eye_noise, rgb.shape)
        z = z * (1 + self.rng.normal(0, CAM_DEPTH_NOISE, z.shape))
        near = np.clip((CAM_MAX - z) / (CAM_MAX - CAM_MIN), 0, 1)
        near[(z < CAM_MIN) | (self.rng.random(z.shape) < CAM_HOLES)] = 0.0
        pix = np.stack([rgb[..., 0], near, rgb[..., 1]], -1).reshape(-1, 3)
        self._cam_last = np.clip(pix[self._cam_pix], 0, 1.5)
        self._cam_rgb, self._cam_near = rgb, near                       # (for the video panel)
        return self._cam_last

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
        if self.harvest:
            return harvest_teacher(self)
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
        return any(np.all(np.abs(b[:2] - self.data.mocap_pos[mc][:2]) < BIN_IN) and b[2] < self.rest_z + 0.012
                   for mc in self.bin_mocaps if mc != self.bin_mocap)

    def in_bin(self):
        """Block resting inside the bin (centre within BIN_IN in x and y, down on the floor)."""
        if self.n_tom > 1:
            return self.tom_in_basket(self._tgt)
        b, c = self.block_pos(), self.bin_pos()
        return bool(np.all(np.abs(b[:2] - c[:2]) < BIN_IN) and b[2] < self.rest_z + 0.012)

    @property
    def attached(self):
        """harvest: the current tomato is still on the vine."""
        return self.att[self._tgt]

    @attached.setter
    def attached(self, v):
        self.att[self._tgt] = v

    def _set_target(self, k):
        """harvest: make tomato k the current one (block_pos, is_grasped, in_bin, ... refer to it)."""
        self._tgt = k
        self.block_q, self.block_geom, self.stalk = self.tom_q[k], self.tom_geoms[k], self.stalks[k]

    def tom_pos(self, k):
        return self.data.qpos[self.tom_q[k]:self.tom_q[k] + 3].copy()

    def grasped_tom(self, k):
        """harvest: tomato k held between both fingers."""
        g, self.block_geom = self.block_geom, self.tom_geoms[k]
        held = self.is_grasped()
        self.block_geom = g
        return held

    def tom_in_basket(self, k):
        """Tomato k resting in the basket (also on top of another tomato in it)."""
        b, c = self.tom_pos(k), self.bin_pos()
        return bool(np.all(np.abs(b[:2] - c[:2]) < BIN_IN) and b[2] < c[2] + self.rest_z + 2 * TOM_R + 0.012)

    def _update_target(self):
        """pickripe: the tomato to work on: the one in the hand, else the ripe one still on the vine nearest the
        hand, else a ripe one not yet in the basket. A function of the present state only (no memory of earlier
        choices), so the teacher's choice is one the memoryless student can make from what it senses now."""
        held = [k for k in range(self.n_tom) if self.grasped_tom(k)]
        if held:
            return self._set_target(held[0])
        on_vine = [k for k in range(self.n_tom) if self.ripe[k] == 2 and self.att[k]]
        if on_vine:
            tcp = self.tcp_pos()
            self._set_target(min(on_vine, key=lambda k: np.linalg.norm(self.tom_pos(k) - tcp)))
            return
        loose = [k for k in range(self.n_tom) if self.ripe[k] == 2 and not self.tom_in_basket(k)]
        if loose:
            self._set_target(loose[0])

    def stalk_point(self):
        """harvest: where the current tomato's centre belongs while on the vine."""
        return self.data.mocap_pos[self.plant_mocap] + np.array([-BRANCH_LEN, TRUSS_Y[self._tgt], -PED_LEN - TOM_R])

    def stalk_force(self, k=None):
        """harvest: how hard the stalk is pulled or bent (N; 0 once picked). The stalk's force on the tomato
        (world frame): sideways part + upward part (the tomato pulled down or away); a tomato pushed up into
        the stalk is not counted. k: which tomato (default: the current one)."""
        k = self._tgt if k is None else k
        if not self.att[k]:
            return 0.0
        d = self.data
        k = (d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id[:d.nefc] == self.stalks[k])
        f = d.efc_force[:d.nefc][k]
        if len(f) != 3:
            return 0.0
        return float(np.hypot(f[0], f[1]) + max(f[2], 0.0))

    def plant_touched(self):
        """harvest: any part of the robot touching the plant."""
        d, m = self.data, self.model
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in self.plant_geoms or g2 in self.plant_geoms:
                other = g2 if g1 in self.plant_geoms else g1
                if m.geom_bodyid[other] in self.robot_bodies:
                    return True
        return False

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
        R_des = self._R_des                                        # z down (harvest: tilted), fixed yaw
        for _ in range(iters):
            d.qpos[:7] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            mujoco.mj_jacSite(m, d, jacp, jacr, self.tcp)
            R = d.site_xmat[self.tcp].reshape(3, 3)
            e_pos = target - d.site_xpos[self.tcp]
            if self.harvest:                                       # (small steps: no jumps between postures)
                n = np.linalg.norm(e_pos)
                if n > 0.02:
                    e_pos = e_pos * 0.02 / n
            E = R_des @ R.T
            e_rot = 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
            Jr = jacr[:, :7]
            if self.harvest:                    # finger direction held firmly, turning about it held loosely
                ax = R[:, 2]
                Pr = np.eye(3) - (1 - YAW_W) * np.outer(ax, ax)
                Jr, e_rot = Pr @ Jr, Pr @ e_rot
            J = np.vstack([jacp[:, :7], Jr])
            e = np.concatenate([e_pos, 0.5 * e_rot])
            JJt = J @ J.T + 1e-4 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            null = np.eye(7) - np.linalg.pinv(J) @ J
            pull = self._null_gain * (self._home - q)
            if self.harvest:                    # and away from joint limits (the tilted hand drifts towards them)
                pull += 0.3 * (np.maximum(self.jnt_lo + 0.3 - q, 0) - np.maximum(q - (self.jnt_hi - 0.3), 0))
            dq += null @ pull
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
            self._last_smell = self.smell_obs()
            out.append(self._last_smell)
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
            if self.n_tom > 1:                  # pickripe: every tomato, riper ones smell stronger
                near = sum(RIPE_SMELL[r] * np.exp(-float(np.sum((tcp + a - self.tom_pos(k)) ** 2)) / (2 * SMELL_SIGMA ** 2))
                           for k, r in enumerate(self.ripe))
            else:
                d2 = float(np.sum((tcp + a - blk) ** 2))
                near = np.exp(-d2 / (2 * SMELL_SIGMA ** 2))
            conc = near * np.exp(self.rng.normal(0, SMELL_FLICKER)) + 0.02
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
        if self.task in ("place", "sort", "smell", "harvest", "pickripe"):
            c = self.bin_pos()
            out += [c, blk - (c + np.array([0, 0, self.rest_z]))]
        if self.harvest:
            out += [[float(self.attached), float(self.plant_touched())]]
        if self.n_tom > 1:                      # which tomatoes are ripe, still on the vine, in the basket
            out += [[float(r == 2) for r in self.ripe], [float(a) for a in self.att],
                    [float(self.tom_in_basket(k)) for k in range(self.n_tom)], [float(self._unripe_picked > 0)]]
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
        if self.harvest:                                # the plant (tomato on its stalk) and the basket
            self._fruit = 0                             # phase 1: always a ripe tomato
            self.bin_mocap = self.bin_mocaps[0]
            c = (BASKET if self.n_tom == 1 else BASKET3) + self.rng.uniform(-SORT_JITTER, SORT_JITTER, 2)
            d.mocap_pos[self.bin_mocap] = [c[0], c[1], 0.0 if self.n_tom == 1 else BASKET3_Z]
            tom = self.rng.uniform(TOM_LO, TOM_HI) if self.n_tom == 1 else self.rng.uniform(TOM_LO3, TOM_HI3)
            d.mocap_pos[self.plant_mocap] = tom + np.array([BRANCH_LEN, 0, PED_LEN + TOM_R])
            d.mocap_pos[self.look_mocap] = [tom[0] + BRANCH_LEN, tom[1], 0.0]
            toms = [tom + np.array([0, TRUSS_Y[k], 0]) for k in range(self.n_tom)]
            self.ripe = [2] * self.n_tom
            if self.n_tom > 1:                          # phase 2: 1 or 2 ripe, the rest green or turning
                n_ripe = int(self.rng.integers(1, self.n_tom)) if self.ripe_max > 1 else 1
                order = self.rng.permutation(self.n_tom)
                for i, k in enumerate(order):
                    self.ripe[k] = 2 if i < n_ripe else int(self.rng.integers(0, 2))
                for k in range(self.n_tom):
                    m.geom_rgba[self.tom_geoms[k]] = RIPE_RGBA[self.ripe[k]]
            for k in range(self.n_tom):
                d.eq_active[self.stalks[k]] = 1
                d.qpos[self.tom_q[k]:self.tom_q[k] + 7] = [*toms[k], 1, 0, 0, 0]
            self.att = [True] * self.n_tom
            self._unripe_picked = 0
            first = min((k for k in range(self.n_tom) if self.ripe[k] == 2),
                        key=lambda k: abs(TRUSS_Y[k])) if self.n_tom > 1 else 0
            self._set_target(first)
            tom = toms[first]
            target = tom[:2]
        if self.task == "place":                        # the bin: somewhere else on the table
            for _ in range(200):
                binc = self.rng.uniform(BIN_LO, BIN_HI)
                if np.linalg.norm(binc - target) > BIN_MIN_GAP:
                    break
            d.mocap_pos[self.bin_mocap] = [binc[0], binc[1], 0.0]
            placed.append(binc)
        if not self.harvest:
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
        if self.harvest:                                    # in front of the plant row, hand pointing at it
            start = self.rng.uniform([0.33, -0.2, 0.10], [0.43, 0.2, 0.28])
            if self.random_start > 0 and self.rng.random() < self.random_start:
                start = tom + self.rng.uniform([-0.13, -0.05, -0.05], [-0.05, 0.05, 0.05])
        mid = held = False
        if self.harvest and self.grasp_start > 0 and self.rng.random() < self.grasp_start:
            mid, held = True, self.rng.random() < 0.5            # open around the tomato, or already closed on it
            start = tom + self.rng.uniform([-0.004, -0.004, -0.004], [0.004, 0.004, 0.004])
        elif (self.n_tom > 1 and self.return_start > 0 and sum(r == 2 for r in self.ripe) >= 2
              and self.rng.random() < self.return_start):
            ripe = [k for k in range(self.n_tom) if self.ripe[k] == 2]
            done = ripe[int(self.rng.integers(len(ripe)))]
            c = self.bin_pos()
            d.eq_active[self.stalks[done]] = 0
            self.att[done] = False
            d.qpos[self.tom_q[done]:self.tom_q[done] + 3] = c + [*self.rng.uniform(-0.02, 0.02, 2), self.rest_z + 0.005]
            self._set_target(next(k for k in ripe if k != done))
            start = c + np.array([0, 0, self.rest_z + HARVEST_CARRY]) + self.rng.uniform(-0.02, 0.02, 3)
        elif self.task in ("lift", "place", "sort", "smell") and self.grasp_start > 0 and self.rng.random() < self.grasp_start:
            mid, held = True, self.rng.random() < 0.5
            start = np.array([target[0], target[1], BLOCK_H]) + self.rng.uniform([-0.008, -0.008, -0.004],
                                                                                   [0.008, 0.008, 0.006])
        if self.harvest:                                    # from the home posture along a straight line (stay
            q = self._home.copy()                           # in the same arm configuration)
            for f in np.linspace(0, 1, 12)[1:]:
                q = self._ik_solve(HOME_HARVEST_TCP + f * (start - HOME_HARVEST_TCP), q, iters=15)
        else:
            q = self._ik_solve(start, self._home, iters=60)
        d.qpos[:7] = q
        d.ctrl[:7] = q
        d.ctrl[7] = self._grip_ctrl = 255.0
        if self.n_tom > 1:                                  # between neighbours: start only part open
            d.ctrl[7] = self._grip_ctrl = (PICK_OPEN + 1) * 127.5
            d.qpos[self.finger_q[0]] = d.qpos[self.finger_q[1]] = 0.04 * self._grip_ctrl / 255
        if mid and not self.harvest:                        # align the block with the fingers
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
        if self.task in ("place", "sort", "smell", "harvest", "pickripe"):   # where the block should end up
            self._goal = self.bin_pos() + np.array([0, 0, self.rest_z])
        self._plant_hits = 0
        self._target = self.tcp_pos()
        self._cmd = np.zeros(3)
        self._hold = 0
        self._last_action = np.zeros(4)
        self._steps = 0
        self._prev_img = None
        self._adapt = None                                  # eyes adapt to this scene from the first look
        self._cam_gain, self._cam_last = None, None
        return self._get_obs(), {}

    def _reward(self):
        if self.n_tom > 1:
            return self._reward_pickripe()
        if self.harvest:
            return self._reward_harvest()
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

    def _reward_harvest(self):
        """Staged, every stage pays more than the one before: reach the tomato (0..1) -> hold it (+1)
        -> pull it off the vine (+1 x pull / SNAP_FORCE) -> picked and held: 3 + 2 x closeness to the
        basket (x/y) -> in the basket, still held (5.5) -> released (6) -> stays there 0.2 s: success (7).
        Minus PLANT_PENALTY while the robot touches the plant. Divided by 7."""
        tcp, tom, c = self.tcp_pos(), self.block_pos(), self.bin_pos()
        d_t = float(np.linalg.norm(tcp - tom))
        held = self.is_grasped()
        r = 1 - np.tanh(5 * d_t)
        if held:
            if self.attached:
                r += 1.0 + float(np.clip(self.stalk_force() / SNAP_FORCE, 0, 1))
            else:
                r = 3.0 + 2.0 * (1 - np.tanh(5 * float(np.linalg.norm(tom[:2] - c[:2]))))
        inside = self.in_bin()
        released = inside and not self.block_touched()
        if inside:
            r = 6.0 if released else 5.5
        self._hold = self._hold + 1 if released else 0
        success = self._hold >= LIFT_HOLD
        if success:
            r = 7.0
        touched = self.plant_touched()
        self._plant_hits += touched
        if touched:
            r -= PLANT_PENALTY
        return r / 7.0, success, d_t

    def _reward_pickripe(self):
        """Phase 2: the phase-1 staged reward (0..7) for the current ripe tomato, on top of 7 for every ripe
        tomato already in the basket, divided by 7 x the number of ripe tomatoes. Holding an unripe tomato
        earns nothing; once an unripe tomato is picked the episode cannot succeed and every step costs 0.3.
        Success: every ripe tomato in the basket, none held, 0.2 s, nothing unripe picked."""
        ripe = [k for k in range(self.n_tom) if self.ripe[k] == 2]
        done = [k for k in ripe if self.tom_in_basket(k)]
        k = self._tgt
        if self.ripe[k] != 2:                                   # holding an unripe one
            r_cur = 0.0
        elif k in done:
            r_cur = 6.0 if not self.block_touched() else 5.5
        else:
            r_cur, _, _ = self._reward_harvest_stage()
        n_done = len(done) - (k in done)
        r = (7.0 * n_done + r_cur) / (7.0 * len(ripe))
        released = len(done) == len(ripe) and not any(self.grasped_tom(j) or self._touching(j) for j in ripe)
        self._hold = self._hold + 1 if (released and self._unripe_picked == 0) else 0
        success = self._hold >= LIFT_HOLD
        if success:
            r = 1.0
        touched = self.plant_touched()
        self._plant_hits += touched
        if touched:
            r -= PLANT_PENALTY / 7.0
        if self._unripe_picked:
            r = -0.3
        return r, success, float(np.linalg.norm(self.tcp_pos() - self.block_pos()))

    def _touching(self, k):
        g, self.block_geom = self.block_geom, self.tom_geoms[k]
        t = self.block_touched()
        self.block_geom = g
        return t

    def _reward_harvest_stage(self):
        """Phase-1 stages for the current tomato, before success / plant terms: (0..6, in_basket, released)."""
        tcp, tom, c = self.tcp_pos(), self.block_pos(), self.bin_pos()
        r = 1 - np.tanh(5 * float(np.linalg.norm(tcp - tom)))
        if self.is_grasped():
            if self.attached:
                r += 1.0 + float(np.clip(self.stalk_force() / SNAP_FORCE, 0, 1))
            else:
                r = 3.0 + 2.0 * (1 - np.tanh(5 * float(np.linalg.norm(tom[:2] - c[:2]))))
        inside = self.in_bin()
        released = inside and not self.block_touched()
        if inside:
            r = 6.0 if released else 5.5
        return r, inside, released

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
            if self.n_tom > 1:
                for k in range(self.n_tom):
                    if self.att[k] and self._grip_ctrl < 40 and self.stalk_force(k) > SNAP_FORCE and self.grasped_tom(k):
                        d.eq_active[self.stalks[k]] = 0     # the stalk snaps at its joint: picked
                        self.att[k] = False
                        self._unripe_picked += self.ripe[k] != 2
            elif self.attached and self._grip_ctrl < 40 and self.stalk_force() > SNAP_FORCE and self.is_grasped():
                d.eq_active[self.stalk] = 0             # the stalk snaps at its joint: picked
                self.attached = False
        self._steps += 1
        self._last_action = a
        if self.n_tom > 1:
            self._update_target()
        r, success, d_t = self._reward()
        truncated = self._steps >= self.max_steps
        info = {"is_success": success, "dist": d_t}
        if self.task in ("lift", "place", "sort", "smell", "harvest", "pickripe"):
            info.update(grasped=self.is_grasped(), lift=self.lift_height(),
                        goal_dist=float(np.linalg.norm(self.block_pos() - self._goal)))
        if self.task in ("place", "sort", "smell", "harvest", "pickripe"):
            info.update(in_bin=self.in_bin())
        if self.harvest:
            info.update(picked=not self.attached, plant_touch_steps=self._plant_hits)
        if self.n_tom > 1:
            ripe = [k for k in range(self.n_tom) if self.ripe[k] == 2]
            info.update(n_ripe=len(ripe), ripe_in_basket=sum(self.tom_in_basket(k) for k in ripe),
                        ripe_picked=sum(not self.att[k] for k in ripe), unripe_picked=self._unripe_picked,
                        ripeness=[RIPENESS[r] for r in self.ripe])
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
            if e == 1 and self.wrist_cam and getattr(self, "_cam_rgb", None) is not None:
                # the camera, for people watching: its full-detail view (top) and what the brain gets (bottom:
                # 16 x 10 colour, and distance, white = near, black = none)
                if getattr(self, "_cam_view_r", None) is None:
                    self._cam_view_r = mujoco.Renderer(self.model, 50, 80)
                vr = self._cam_view_r
                vr.update_scene(self.data, camera="wrist_cam", scene_option=self._cam_opt)
                vr.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                full = vr.render().astype(np.float64) / 255.0
                vr.enable_depth_rendering()
                vr.update_scene(self.data, camera="wrist_cam", scene_option=self._cam_opt)
                zf = vr.render().astype(np.float64)
                vr.disable_depth_rendering()
                nf = np.clip((CAM_MAX - zf) / (CAM_MAX - CAM_MIN), 0, 1) * (zf >= CAM_MIN)
                out[2:52, 2:82] = full
                out[2:52, 86:166] = nf[..., None]
                q = 4
                cam = np.repeat(np.repeat(np.clip(self._cam_rgb, 0, 1), q, 0), q, 1)
                dep = np.repeat(np.repeat(np.repeat(self._cam_near[..., None], 3, 2), q, 0), q, 1)
                out[54:54 + CAM_H * q, 2:2 + CAM_W * q] = cam[:h - 52]
                out[54:54 + CAM_H * q, 86:86 + CAM_W * q] = dep[:h - 52]
                continue
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
        self._renderer.update_scene(self.data, camera=self._cam, scene_option=self._vopt)
        if self.harvest:                    # no shadows: hundreds of leaves made each frame ~5x slower to draw
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        frame = self._renderer.render().copy()
        eye = self.eye_view()
        frame[6:6 + eye.shape[0], 6:6 + eye.shape[1]] = eye
        if self.task == "smell":
            frame = self._smell_overlay(frame)
        if self.harvest:
            frame = self._harvest_overlay(frame)
        return frame

    def _harvest_overlay(self, frame):
        """For people watching (the brain never sees this): what stage the tomato is at."""
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(frame)
        dr = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default(15)
        except TypeError:
            font = ImageFont.load_default()
        stage = ("in the basket" if self.in_bin() else "on the vine" if self.attached else
                 "picked, carrying" if self.is_grasped() else "picked, falling / dropped")
        if self.n_tom > 1:                      # label each tomato with its ripeness
            ripe = [k for k in range(self.n_tom) if self.ripe[k] == 2]
            shown = [k for k in range(self.n_tom) if self.att[k]]
            pts = {k: self._project(self.tom_pos(k)) for k in shown}
            for i, k in enumerate(sorted(shown, key=lambda k: pts[k][0])):   # a label column left of the truss
                x0 = min(p[0] for p in pts.values()) - 95
                y0 = int(np.mean([p[1] for p in pts.values()])) - 30 + 22 * i
                col = ((60, 150, 40), (220, 120, 30), (200, 30, 20))[self.ripe[k]]
                dr.line((x0 + 62, y0 + 8, pts[k][0] - 6, pts[k][1]), fill=col, width=1)
                dr.text((x0, y0), RIPENESS[self.ripe[k]], fill=col, font=font, stroke_width=2, stroke_fill=(255, 255, 255))
            stage = (f"ripe in the basket {sum(self.tom_in_basket(k) for k in ripe)}/{len(ripe)}"
                     + (f"   UNRIPE PICKED: {self._unripe_picked}" if self._unripe_picked else ""))
        x, y = self._project(self.bin_pos() + np.array([0, 0, 0.05]))
        dr.text((x - 25, y - 22), "BASKET", fill=(150, 90, 40), font=font, stroke_width=2, stroke_fill=(255, 255, 255))
        dr.rectangle((4, 480 - 50, 330, 480 - 4), fill=(20, 22, 28))
        dr.text((10, 480 - 46), stage if self.n_tom > 1 else f"tomato: {stage}", fill=(240, 240, 240), font=font)
        dr.text((10, 480 - 26), f"robot touching the plant: {self._plant_hits} steps", fill=(200, 200, 200), font=font)
        return np.asarray(img)

    def _project(self, p):
        """Pixel (x, y) of world point p in the overview camera image (640 x 480)."""
        cam = self.model.camera(self._cam).id
        pos, R = self.data.cam_xpos[cam], self.data.cam_xmat[cam].reshape(3, 3)
        v = R.T @ (np.asarray(p) - pos)                      # camera frame: x right, y up, -z forward
        f = 240 / np.tan(np.radians(self.model.cam_fovy[cam]) / 2)
        return int(320 + f * v[0] / -v[2]), int(240 - f * v[1] / -v[2])

    def _smell_overlay(self, frame):
        """For people watching (the brain never sees this): which fruit it is, which bin is which, and
        what the two antennae report right now."""
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(frame)
        dr = ImageDraw.Draw(img)
        try:
            font, small = ImageFont.load_default(15), ImageFont.load_default(11)
        except TypeError:                                      # older Pillow
            font = small = ImageFont.load_default()
        for k, (name, col) in enumerate((("RIPE", (60, 170, 60)), ("ROTTEN", (150, 90, 40)))):
            x, y = self._project(self.data.mocap_pos[self.bin_mocaps[k]] + np.array([0, 0, 0.03]))
            dr.text((x - 22, y - 26), name, fill=col, font=small, stroke_width=2, stroke_fill=(255, 255, 255))
        fruit = ("RIPE", "ROTTEN")[self._fruit]
        dest = ("left", "right")[self._fruit]
        x0, y0 = 8, 480 - 112
        dr.rectangle((x0 - 4, y0 - 4, x0 + 318, 480 - 4), fill=(20, 22, 28))
        dr.text((x0, y0), f"fruit: {fruit}  (-> {dest} bin)   looks identical", fill=(240, 240, 240), font=font)
        s = getattr(self, "_last_smell", None)
        G = self.smell_resp.shape[0]
        dr.text((x0, y0 + 22), "what the antennae smell (53 receptor types)", fill=(170, 170, 170), font=small)
        if s is not None:
            for a in range(2):                                 # one row per antenna, one cell per receptor type
                for g in range(G):
                    v = float(np.clip(s[a * G + g], -1, 1))
                    c = (int(40 + 200 * v), int(40 + 120 * v), 50) if v >= 0 else (60, 50, int(40 - 200 * v))
                    dr.rectangle((x0 + 18 + g * 5.6, y0 + 38 + a * 9, x0 + 22 + g * 5.6, y0 + 45 + a * 9), fill=c)
                dr.text((x0, y0 + 36 + a * 9), "LR"[a], fill=(170, 170, 170), font=small)
            ester = float(np.mean([s[a * G + g] for a in range(2) for g in self._smell_groups[0]]))
            acid = float(np.mean([s[a * G + g] for a in range(2) for g in self._smell_groups[1]]))
            for k, (lab, v, col) in enumerate((("fruity-ester sensors", ester, (90, 200, 90)),
                                              ("acid sensors (rot)", acid, (210, 140, 60)))):
                y = y0 + 62 + k * 18
                dr.text((x0, y), lab, fill=(200, 200, 200), font=small)
                dr.rectangle((x0 + 140, y + 2, x0 + 300, y + 11), outline=(90, 90, 90))
                dr.rectangle((x0 + 140, y + 2, x0 + 140 + int(160 * max(0.0, min(1.0, v))), y + 11), fill=col)
        return np.asarray(img)

    def close(self):
        if getattr(self, "_cam_view_r", None) is not None:
            self._cam_view_r.close()
            self._cam_view_r = None
        if self._cam_r is not None:
            self._cam_r.close()
            self._cam_r = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def panda_controller(env, obs=None):
    """Hand-written solution using the TRUE block position (the brain never gets it)."""
    if env.harvest:
        return harvest_teacher(env)
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


def harvest_teacher(env):
    """Harvest teacher (TRUE tomato, plant and basket positions; training targets only). Stateless, the hand
    tilted towards the plant (APPROACH): on the vine, not held -> line up PRE_GRASP back along the approach
    line (backing out first if close but off the line), go in along it with the hand open, close; held on the
    vine -> pull back and down until the stalk snaps; picked and held -> carry over the basket at HARVEST_CARRY
    and open (it drops in); in the basket -> back away, open; dropped elsewhere -> wait, open."""
    tcp, tom = env.tcp_pos(), env.block_pos()
    a = np.zeros(4)
    open_a = 1.0 if env.n_tom == 1 else PICK_OPEN          # between neighbours: open only part way
    if env.n_tom > 1 and env.ripe[env._tgt] != 2:          # (holding an unripe one by mistake: let go)
        a[3] = open_a
        a[:3] = np.clip((tcp + np.array([-0.03, 0, 0.03]) - tcp) / STEP * 0.6, -1, 1)
        return a
    if env.n_tom > 1 and env.attached and not env.is_grasped():
        # the head eye's facets are ~11 deg apart and the three tomatoes ~6 deg apart from there: from a distance
        # the truss is one blur. So the teacher first brings the wrist eye close to the middle of the truss
        # (the same move whichever tomato is ripe), and only then goes for a ripe one. Its far moves give
        # nothing away that the student cannot see.
        mid = env.data.mocap_pos[env.plant_mocap] + np.array([-BRANCH_LEN, 0, -PED_LEN - TOM_R])
        if np.linalg.norm(tcp - mid) > LOOK_NEAR:
            a[3] = open_a
            a[:3] = np.clip((mid - LOOK_BACK * APPROACH - tcp) / STEP * 0.6, -1, 1)
            return a
    if env.attached:
        if env.is_grasped() and env._grip_ctrl < 40:
            target = tcp + np.array([-0.03, 0, -0.03])         # pull it off
            a[3] = -1.0
        else:
            rel = tom - tcp
            ahead = float(rel @ APPROACH)                      # along the approach line
            lat = float(np.linalg.norm(rel - ahead * APPROACH))   # off it
            is_open = env._grip_ctrl > (200 if env.n_tom == 1 else 90)
            if ahead < 0.035 and lat > 0.015 and is_open:      # close but off the line: back out first
                target = tcp - (PRE_GRASP - ahead) * APPROACH
                a[3] = open_a
            elif ahead >= 0.035 and lat > 0.012 and is_open:   # line up behind it on the approach line
                target = tom - PRE_GRASP * APPROACH
                a[3] = open_a
            elif (np.linalg.norm(rel) > (0.005 if env.n_tom == 1 else 0.009) or np.linalg.norm(env.tcp_vel()) > 0.03) and is_open:
                target = tom.copy()                            # straight in, open
                a[3] = open_a
            else:
                target = tom.copy()                            # there: close
                a[3] = -1.0
    elif env.is_grasped():
        c = env.bin_pos()
        want = np.array([c[0], c[1], c[2] + env.rest_z + HARVEST_CARRY])
        a[3] = -1.0
        if np.linalg.norm(tom[:2] - c[:2]) < 0.012:
            a[3] = 1.0                                         # over the basket: let go (wide open)
        target = want - (tom - tcp)
        a[:3] = np.clip((target - tcp) / STEP * 0.6, -0.5, 0.5)   # carry gently (a round fruit can slip)
        return a
    elif env.in_bin():
        target = tcp + np.array([-0.03, 0, 0.03])
        a[3] = 1.0
    else:
        target = tcp.copy()
        a[3] = 1.0
    a[:3] = np.clip((target - tcp) / STEP * 0.6, -1, 1)
    return a
