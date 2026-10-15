"""Keep a laptop (especially a fanless/8 GB Mac M1) cool and responsive while training.

What it does:
  1. limit_threads()   caps PyTorch / BLAS / OpenMP threads (default 2 of 8 cores)
  2. lower_priority()  'nice' the process so the Mac stays usable
  3. SafetyCallback    during training:
       - duty cycle: work N seconds, then sleep M seconds (lets the chip cool)
       - macOS thermal check via `pmset -g therm`; long cool-down if throttling
       - stops (and saves) if free RAM gets low
       - stops if on battery below a threshold
       - stops after a wall-clock time limit
     Training is always saved before stopping, so you can --resume.

Nothing here is needed on Kaggle; it is switched off by the 'kaggle' profile.
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
import time

import psutil
from stable_baselines3.common.callbacks import BaseCallback

IS_MAC = platform.system() == "Darwin"


def limit_threads(n: int):
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[var] = str(n)
    import torch
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set


def lower_priority(level: int = 10):
    try:
        os.nice(level)
    except OSError:
        pass


def mac_thermal_status() -> dict:
    """Parse `pmset -g therm`. Returns {'warning': int, 'cpu_limit': int} (best effort)."""
    out = {"warning": 0, "cpu_limit": 100, "raw": ""}
    if not IS_MAC:
        return out
    try:
        txt = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return out
    out["raw"] = txt.strip()
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", txt)
    if m:
        out["cpu_limit"] = int(m.group(1))
    m = re.search(r"[Tt]hermal warning level (?:has been set to|=)\s*(\d+)", txt)
    if m:
        out["warning"] = int(m.group(1))
    return out


class SafetyCallback(BaseCallback):
    def __init__(self, save_fn, work_s=45, rest_s=15, max_minutes=None, min_free_gb=1.0,
                 min_battery=30, check_every_s=10, cooldown_s=120, verbose=1):
        super().__init__(verbose)
        self.save_fn = save_fn
        self.work_s, self.rest_s = work_s, rest_s
        self.max_s = None if max_minutes is None else max_minutes * 60
        self.min_free_gb = min_free_gb
        self.min_battery = min_battery
        self.check_every_s = check_every_s
        self.cooldown_s = cooldown_s
        self._t_start = self._t_work = self._t_check = None
        self.rest_total = 0.0
        self.stop_reason = None

    def _on_training_start(self):
        now = time.time()
        self._t_start = self._t_work = self._t_check = now

    def _stop(self, reason):
        self.stop_reason = reason
        print(f"\n[safety] stopping: {reason}. Saving checkpoint...")
        self.save_fn()
        return False

    def _sleep(self, s, why):
        if self.verbose:
            print(f"[safety] {why}: resting {s:.0f}s")
        time.sleep(s)
        self.rest_total += s

    def _on_step(self) -> bool:
        now = time.time()
        if now - self._t_check < self.check_every_s:
            return True
        self._t_check = now

        if self.max_s is not None and now - self._t_start - self.rest_total > self.max_s:
            return self._stop("time limit reached")

        free_gb = psutil.virtual_memory().available / 1e9
        if free_gb < self.min_free_gb:
            return self._stop(f"low memory ({free_gb:.1f} GB free)")

        batt = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
        if batt is not None and not batt.power_plugged and batt.percent < self.min_battery:
            return self._stop(f"on battery at {batt.percent:.0f}%")

        th = mac_thermal_status()
        if th["warning"] > 0 or th["cpu_limit"] < 100:
            self._sleep(self.cooldown_s, f"Mac is thermally throttling ({th['warning']=}, {th['cpu_limit']=})")
            self._t_work = time.time()
            return True

        if self.work_s and now - self._t_work > self.work_s:
            self._sleep(self.rest_s, "duty cycle")
            self._t_work = time.time()
        return True
