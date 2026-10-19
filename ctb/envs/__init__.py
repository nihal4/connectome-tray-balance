"""Environment registry. Use make_env(name, seed, **overrides).

    tray          TrayBalance: keep a ball on a flat tray (easy, solved in ~100k steps)
    maze_fixed3   tilt-maze puzzle, one fixed 3x3 maze
    maze3         random 3x3 mazes (new puzzle every episode)
    maze3h        random 3x3 mazes with holes
    maze4         random 4x4 mazes with holes
    maze5         random 5x5 mazes with holes and some loops
    *_vision      same task, but seen through a 16x16 wrist camera (tray_vision,
                  maze_fixed3_vision, maze3_vision, maze3h_vision)
    drone_reach   fly-brain drone with a compound eye: find a target ball and hover on it
    panda_reach   Franka Panda arm + gripper, hand-space control, two compound eyes (head + wrist):
                  bring the gripper to the yellow block (pre-grasp pose) and hold still
    panda_lift    same robot + touch: grasp the yellow block and lift it 8 cm
    panda_place   pick and place: carry the yellow block into a red bin and let go
"""
import numpy as np

from .labyrinth import LEVELS, LabyrinthEnv, planner_controller
from .drone import FlyDroneEnv, drone_controller
from .panda import FlyPandaEnv, panda_controller
from .tray_balance import HOME_QPOS, TrayBalanceEnv, pd_controller

ENVS = {"tray": (TrayBalanceEnv, {})}
ENVS.update({name: (LabyrinthEnv, kw) for name, kw in LEVELS.items()})
# Camera versions: the policy sees a 16x16 wrist-camera image (this frame + previous)
# instead of the ball position, wall map and goal; plus its own joints.
ENVS["tray_vision"] = (TrayBalanceEnv, {"vision": True})
for _name in ["maze_fixed3", "maze3", "maze3h"]:
    ENVS[_name + "_vision"] = (LabyrinthEnv, {**LEVELS[_name], "vision": True})
ENVS["drone_reach"] = (FlyDroneEnv, {"task": "reach"})
ENVS["panda_reach"] = (FlyPandaEnv, {"task": "reach"})
ENVS["panda_lift"] = (FlyPandaEnv, {"task": "lift"})
ENVS["panda_place"] = (FlyPandaEnv, {"task": "place"})


def build_env(name: str = "tray", seed: int | None = None, **overrides):
    cls, kw = ENVS[name]
    return cls(seed=seed, **{**kw, **overrides})


def make_env(name: str = "tray", seed: int = 0, **overrides):
    def _thunk():
        return build_env(name, seed, **overrides)
    return _thunk


def scripted_controller(env, obs):
    """Hand-written solution for any env (proves the task is solvable)."""
    if isinstance(env, FlyPandaEnv):
        return panda_controller(env, obs)
    if isinstance(env, FlyDroneEnv):
        return drone_controller(env, obs)
    if isinstance(env, LabyrinthEnv):
        return planner_controller(env, obs)
    if getattr(env, "vision", False):         # camera obs has no ball numbers: use the true state
        bp, bv = env._ball_in_tray()
        obs = np.zeros(24)
        obs[10:12], obs[13:15] = bp[:2] * 10, bv[:2]
    return pd_controller(obs)


__all__ = ["scripted_controller", "pd_controller", "ENVS", "build_env", "make_env", "TrayBalanceEnv", "LabyrinthEnv", "planner_controller", "HOME_QPOS", "FlyDroneEnv", "drone_controller", "FlyPandaEnv", "panda_controller"]
