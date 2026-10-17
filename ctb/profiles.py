"""Hardware profiles. Pick with --profile. Any value can be overridden on the command line."""

# Kept here (not imported from ctb.envs) so scripts can parse arguments before
# numpy/torch load, which lets the thread limits take effect.
ENV_NAMES = ["tray", "maze_fixed3", "maze3", "maze3h", "maze4", "maze5",
             "tray_vision", "maze_fixed3_vision", "maze3_vision", "maze3h_vision", "drone_reach", "panda_reach"]

PROFILES = {
    # Mac M1 8 GB: small, slow, cool. Meant for checking that things learn, not final results.
    "mac": dict(
        n_envs=2, vec="dummy", threads=2, nice=10,
        total_steps=200_000, n_steps=1024, batch_size=256, n_epochs=5,
        safety=True, work_s=45, rest_s=15, max_minutes=30, min_free_gb=1.0,
        eval_every=20_000, eval_episodes=5,
    ),
    # Kaggle CPU session (4 vCPU, ~30 GB RAM). The GPU barely helps here because the
    # physics runs on CPU and the network is small, so save your GPU quota.
    "kaggle": dict(
        n_envs=4, vec="subproc", threads=4, nice=0,
        total_steps=2_000_000, n_steps=1024, batch_size=512, n_epochs=10,
        safety=False, work_s=0, rest_s=0, max_minutes=11 * 60, min_free_gb=1.0,
        eval_every=50_000, eval_episodes=10,
    ),
    # Always-on Linux desktop (e.g. i3-8100, 4 cores, 12 GB): full speed, no time limit,
    # but stops/cools down on low RAM or a very hot CPU. Runs for days with --resume.
    "desktop": dict(
        n_envs=4, vec="subproc", threads=4, nice=5,
        total_steps=2_000_000, n_steps=1024, batch_size=512, n_epochs=10,
        safety=True, work_s=0, rest_s=0, max_minutes=0, min_free_gb=1.0,
        eval_every=50_000, eval_episodes=10,
    ),
    # Tiny run to check the pipeline end to end (~1-2 minutes).
    "smoke": dict(
        n_envs=2, vec="dummy", threads=2, nice=10,
        total_steps=4096, n_steps=512, batch_size=256, n_epochs=2,
        safety=True, work_s=45, rest_s=5, max_minutes=5, min_free_gb=0.5,
        eval_every=2048, eval_episodes=2,
    ),
}
