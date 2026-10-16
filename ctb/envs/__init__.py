"""Environment registry. Use make_env(name, seed, **overrides).

    tray          TrayBalance: keep a ball on a flat tray (easy, solved in ~100k steps)
    maze_fixed3   tilt-maze puzzle, one fixed 3x3 maze
    maze3         random 3x3 mazes (new puzzle every episode)
    maze3h        random 3x3 mazes with holes
    maze4         random 4x4 mazes with holes
    maze5         random 5x5 mazes with holes and some loops
"""
from .labyrinth import LEVELS, LabyrinthEnv, planner_controller
from .tray_balance import HOME_QPOS, TrayBalanceEnv, pd_controller

ENVS = {"tray": (TrayBalanceEnv, {})}
ENVS.update({name: (LabyrinthEnv, kw) for name, kw in LEVELS.items()})


def build_env(name: str = "tray", seed: int | None = None, **overrides):
    cls, kw = ENVS[name]
    return cls(seed=seed, **{**kw, **overrides})


def make_env(name: str = "tray", seed: int = 0, **overrides):
    def _thunk():
        return build_env(name, seed, **overrides)
    return _thunk


def scripted_controller(env, obs):
    """Hand-written solution for any env (proves the task is solvable)."""
    if isinstance(env, LabyrinthEnv):
        return planner_controller(env, obs)
    return pd_controller(obs)


__all__ = ["scripted_controller", "pd_controller", "ENVS", "build_env", "make_env", "TrayBalanceEnv", "LabyrinthEnv", "planner_controller", "HOME_QPOS"]
