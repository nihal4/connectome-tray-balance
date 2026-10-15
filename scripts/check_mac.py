"""Quick health check before training. Safe to run on a Mac M1 8 GB (~1 minute, 2 threads).

Checks, in order:
  1. packages import, versions, threads, free RAM, battery, thermal state
  2. the MuJoCo tray environment steps and reports its speed
  3. a hand-written PD controller can balance the ball (proves the task is solvable)
  4. the connectome subset loads and the brain-shaped network runs forward + backward
  5. estimated training speed on this machine

Usage:
    python scripts/check_mac.py
    python scripts/check_mac.py --subset data/subset_synthetic_N300.npz   # before downloading
"""
import argparse
import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ctb.safety import limit_threads, lower_priority, mac_thermal_status  # noqa: E402

limit_threads(2)
lower_priority(10)

OK, BAD = "  [ok]", "  [!!]"


def section(t):
    print(f"\n== {t}")


def pd_controller(obs):
    """Tilt the tray against the ball: uses wrist pitch and roll only."""
    import numpy as np
    bx, by = obs[10] / 10, obs[11] / 10
    vx, vy = obs[13], obs[14]
    a = np.zeros(5)
    a[3] = -(3.0 * bx + 1.5 * vx)
    a[4] = 3.0 * by + 1.5 * vy
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="data/subset_flywire783_N1000.npz")
    args = ap.parse_args()

    section("System")
    import numpy as np
    import psutil
    print(f"{OK} {platform.system()} {platform.machine()}, python {platform.python_version()}")
    vm = psutil.virtual_memory()
    print(f"{OK} RAM free {vm.available / 1e9:.1f} / {vm.total / 1e9:.1f} GB")
    batt = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    if batt:
        print(f"{OK if batt.power_plugged else BAD} battery {batt.percent:.0f}% "
              f"({'plugged in' if batt.power_plugged else 'ON BATTERY - plug in before training'})")
    th = mac_thermal_status()
    if platform.system() == "Darwin":
        print(f"{OK if th['warning'] == 0 else BAD} thermal warning level {th['warning']}, CPU limit {th['cpu_limit']}%")
    try:
        import gymnasium
        import mujoco
        import stable_baselines3
        import torch
        print(f"{OK} torch {torch.__version__} (threads={torch.get_num_threads()}), mujoco {mujoco.__version__}, "
              f"gymnasium {gymnasium.__version__}, sb3 {stable_baselines3.__version__}")
    except ImportError as e:
        print(f"{BAD} missing package: {e}. Run: pip install -r requirements.txt")
        return

    section("Environment")
    from ctb.envs import TrayBalanceEnv
    env = TrayBalanceEnv(seed=0)
    obs, _ = env.reset()
    t0, n = time.time(), 0
    while n < 2000:
        obs, r, te, tr, _ = env.step(env.action_space.sample())
        n += 1
        if te or tr:
            env.reset()
    sps = n / (time.time() - t0)
    print(f"{OK} obs dim {obs.shape[0]}, action dim {env.action_space.shape[0]}, {sps:.0f} env steps/s (1 core)")

    section("Task is solvable (hand-written PD controller vs doing nothing)")
    for name, ctrl in [("do nothing", lambda o: np.zeros(5)), ("PD controller", pd_controller)]:
        lens = []
        for ep in range(5):
            obs, _ = env.reset(seed=ep)
            for t in range(500):
                obs, r, te, tr, _ = env.step(ctrl(obs))
                if te or tr:
                    break
            lens.append(t + 1)
        print(f"{OK} {name:14s} survives {np.mean(lens):5.0f} / 500 steps")
    env.close()

    section("Connectome network")
    if not os.path.exists(args.subset):
        print(f"{BAD} {args.subset} not found.")
        print("      Run: python scripts/download_connectome.py && python scripts/build_subset.py --n 1000")
        print("      Or test offline: python scripts/build_subset.py --synthetic --n 300 and pass --subset")
        return
    from ctb.policies import ConnectomeCore
    d = np.load(args.subset)
    print(f"{OK} {args.subset}: {int(d['n'])} neurons, {len(d['pre']):,} connections, source={d['source']}")
    core = ConnectomeCore(obs.shape[0], args.subset)
    x = torch.randn(256, obs.shape[0])
    t0 = time.time()
    for _ in range(5):
        y = core(x)
        y.pow(2).mean().backward()
    ms = (time.time() - t0) / 5 * 1000
    print(f"{OK} forward+backward on batch 256: {ms:.0f} ms, output {tuple(y.shape)}, "
          f"activity std {y.std().item():.3f}")
    print(f"{OK} trainable params {sum(p.numel() for p in core.parameters()):,}")
    with torch.no_grad():
        x2 = torch.randn(2, obs.shape[0])
        t0 = time.time()
        for _ in range(50):
            core(x2)
        infer_ms = (time.time() - t0) / 50 * 1000
    print(f"{OK} acting (batch 2): {infer_ms:.1f} ms per control step")

    section("Estimate (mac profile: 2 envs, 5 epochs, batch 256)")
    rollout_s = 1024 * (2 / sps + infer_ms / 1000)   # env steps + policy inference
    update_s = 5 * (2048 / 256) * ms / 1000
    per_200k = (rollout_s + update_s) * (200_000 / 2048) / 60
    print(f"{OK} roughly {per_200k:.0f} min of compute per 200k steps "
          f"(+25% rest time from the duty cycle). Rough estimate only.")
    print("\nAll checks done. Next: python scripts/train.py --policy connectome --profile smoke")


if __name__ == "__main__":
    main()
