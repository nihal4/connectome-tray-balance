#!/usr/bin/env bash
# Runs the job in ~/flybrain/job.env. Always --resume: after a reboot or power cut it
# continues from the last checkpoint (or starts fresh if there is none).
# Queue (fly queue add ...): when a job FINISHES, the next queued job is set and started here; a stop, a failure
# or a reboot never advances the queue (training never starts by itself).
set -euo pipefail
ROOT="$HOME/flybrain"
mkdir -p "$ROOT/logs"
while true; do
    source "$ROOT/job.env"
    cd "$ROOT/repo"
    "$ROOT/.venv/bin/python" -u scripts/train.py --env "$ENV" --policy connectome --subset "$BRAIN" \
        --profile desktop --total_steps "$STEPS" --seed "${SEED:-0}" --runs_dir "$ROOT/runs" --name "$NAME" \
        --resume $EXTRA >> "$ROOT/logs/$NAME.log" 2>&1
    LAST=$(tail -n 5 "$ROOT/logs/$NAME.log")
    grep -q "^\[train\] finished\." <<< "$LAST" || exit 0                      # stopped early: do not advance
    Q="$ROOT/queue.txt"
    [ -s "$Q" ] || exit 0
    LINE=$(head -n 1 "$Q"); sed -i 1d "$Q"
    IFS='|' read -r QENV QSTEPS QEXTRA <<< "$LINE"
    echo "[queue] next: $QENV $QSTEPS $QEXTRA" >> "$ROOT/logs/$NAME.log"
    "$ROOT/repo/scripts/desktop/fly" job "$QENV" "$QSTEPS" "$QEXTRA"
done
