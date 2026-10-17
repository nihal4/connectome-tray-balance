#!/usr/bin/env bash
# Runs the job in ~/flybrain/job.env. Always --resume: after a reboot or power cut it
# continues from the last checkpoint (or starts fresh if there is none).
set -euo pipefail
ROOT="$HOME/flybrain"
source "$ROOT/job.env"
cd "$ROOT/repo"
mkdir -p "$ROOT/logs"
exec "$ROOT/.venv/bin/python" -u scripts/train.py --env "$ENV" --policy connectome --subset "$BRAIN" \
    --profile desktop --total_steps "$STEPS" --seed "${SEED:-0}" --runs_dir "$ROOT/runs" --name "$NAME" \
    --resume $EXTRA >> "$ROOT/logs/$NAME.log" 2>&1
