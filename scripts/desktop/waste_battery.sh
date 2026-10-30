#!/usr/bin/env bash
# The waste-station test battery (scripts/waste_eval.py) on every panda_waste run, one after another, after the
# training queue is done (waits while training runs; never starts training). Results: ~/flybrain/waste_results/
#   bash ~/flybrain/repo/scripts/desktop/waste_battery.sh [N=200]
set -uo pipefail
ROOT="$HOME/flybrain"; N="${1:-200}"; OUT="$ROOT/waste_results"; mkdir -p "$OUT"
while systemctl --user is-active -q flybrain-train; do sleep 60; done
cd "$ROOT/repo"
for d in "$ROOT"/runs/panda_waste-*; do
    r=$(basename "$d")
    [ -f "$OUT/$r.json" ] || nice -n 10 "$ROOT/.venv/bin/python" scripts/waste_eval.py "$d" "$OUT/$r.json" "$N" 2>&1 \
        | grep -v "Warn\|WARNING\|EGL" | tee -a "$OUT/battery.log"
    if [ -f "$d/imitate_best_policy.pt" ] && [ ! -f "$OUT/$r-imitation.json" ]; then
        BRAIN=imitation nice -n 10 "$ROOT/.venv/bin/python" scripts/waste_eval.py "$d" "$OUT/$r-imitation.json" "$N" 2>&1 \
            | grep -v "Warn\|WARNING\|EGL" | tee -a "$OUT/battery.log"
    fi
done
echo "battery done: $OUT" | tee -a "$OUT/battery.log"
