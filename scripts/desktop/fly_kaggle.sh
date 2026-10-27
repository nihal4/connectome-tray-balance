#!/usr/bin/env bash
# Kaggle GPU jobs for the fly brain.        fly kaggle help
#
# Security: the Kaggle key lives OUTSIDE the repo, in ~/.config/flybrain/secrets.env (only you can read it,
# chmod 600). It is never written into the code, never put on a command line (other users could see it in
# `ps`), and only handed to the kaggle program's environment. A git hook blocks commits that contain it.
set -euo pipefail
ROOT="$HOME/flybrain"
REPO="$ROOT/repo"
PY="$ROOT/.venv/bin/python"
KAGGLE="$ROOT/.venv/bin/kaggle"
SEC="${FLY_SECRETS:-$HOME/.config/flybrain/secrets.env}"
OUTD="$ROOT/kaggle"

load_secrets() {
  [ -f "$SEC" ] || { echo "no Kaggle key yet: run  fly kaggle setup"; exit 1; }
  [ "$(stat -c %a "$SEC")" = 600 ] || { chmod 600 "$SEC"; echo "(set $SEC to owner-only, 600)"; }
  while IFS='=' read -r k v; do                 # read only the known names: the file is data, not a script
    case "$k" in KAGGLE_USERNAME|KAGGLE_KEY|KAGGLE_API_TOKEN) export "$k=$v" ;; esac
  done < "$SEC"
  [ -n "${KAGGLE_USERNAME:-}" ] || { echo "KAGGLE_USERNAME missing in $SEC (run fly kaggle setup)"; exit 1; }
}

need_cli() { [ -x "$KAGGLE" ] || "$PY" -m pip install -q kaggle; }

install_hook() {
  local H="$REPO/.git/hooks/pre-commit"
  if [ -f "$H" ] && ! grep -q "flybrain-secret-guard" "$H"; then mv "$H" "$H.before-flybrain"; fi
  cp "$REPO/scripts/desktop/pre-commit-secrets" "$H" && chmod 755 "$H"
  echo "git hook installed: commits containing the key or a secret file are blocked"
}

case "${1:-help}" in
  setup)
    # fly kaggle setup                   type the username and key (hidden)
    # fly kaggle setup ~/Downloads/kaggle.json   take them from the downloaded file (then deletes it)
    need_cli
    mkdir -p "$(dirname "$SEC")" && chmod 700 "$(dirname "$SEC")"
    if [ -n "${2:-}" ]; then
      [ -f "$2" ] || { echo "no file $2"; exit 1; }
      U=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["username"])' "$2")
      K=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["key"])' "$2")
    else
      read -r -p "Kaggle username: " U
      echo "Kaggle API key or token (kaggle.com > Settings > API). Typing is hidden:"
      read -r -s K; echo
    fi
    [ -n "$U" ] && [ -n "$K" ] || { echo "nothing saved"; exit 1; }
    ( umask 077
      { echo "KAGGLE_USERNAME=$U"
        case "$K" in KGAT_*) echo "KAGGLE_API_TOKEN=$K" ;; *) echo "KAGGLE_KEY=$K" ;; esac; } > "$SEC" )
    unset K
    if [ -n "${2:-}" ]; then shred -u "$2" 2>/dev/null || rm -f "$2"; echo "deleted $2 (the key is now only in $SEC)"; fi
    install_hook
    load_secrets
    if "$KAGGLE" kernels list --mine --page-size 1 >/dev/null 2>&1; then echo "key works: saved in $SEC (owner-only)"
    else echo "saved in $SEC, but Kaggle refused it: check the username/key (fly kaggle setup again)"; exit 1; fi ;;
  mjx)
    # fly kaggle mjx ["--batches 64,256 --impl warp"]   push the MJX feasibility test (private notebook, T4 GPU)
    need_cli; load_secrets
    D="$OUTD/push-mjx"; rm -rf "$D"
    "$PY" "$REPO/scripts/kaggle/pack.py" --job mjx --user "$KAGGLE_USERNAME" --out "$D" --args "${2:-}"
    "$KAGGLE" kernels push -p "$D"
    echo "running on Kaggle (about 20-60 min). Check:  fly kaggle status    when complete:  fly kaggle pull" ;;
  status)
    need_cli; load_secrets
    "$KAGGLE" kernels status "$KAGGLE_USERNAME/flybrain-${2:-mjx}" ;;
  pull)
    need_cli; load_secrets
    J="${2:-mjx}"; mkdir -p "$OUTD/$J"
    "$KAGGLE" kernels output "$KAGGLE_USERNAME/flybrain-$J" -p "$OUTD/$J" >/dev/null
    S=$(find "$OUTD/$J" -name summary.txt | head -1)
    if [ -n "$S" ]; then cat "$S"; echo; echo "full report: $(dirname "$S")/report.json"
    else echo "no summary yet; the log is in $OUTD/$J:"; ls "$OUTD/$J"; tail -n 30 "$OUTD/$J"/*.log 2>/dev/null || true; fi ;;
  forget)
    # remove the saved key from this PC (also revoke it on kaggle.com > Settings > API)
    [ -f "$SEC" ] && { shred -u "$SEC" 2>/dev/null || rm -f "$SEC"; echo "deleted $SEC"; } || echo "no key saved" ;;
  *) cat <<HELP
fly kaggle setup [kaggle.json]   save your Kaggle username + key on this PC (outside the repo, owner-only)
fly kaggle mjx ["ARGS"]          run the MJX GPU feasibility test on Kaggle (private notebook, T4)
fly kaggle status [JOB]          is it still running?
fly kaggle pull [JOB]            download the result and show the summary
fly kaggle forget                delete the saved key from this PC
HELP
  ;;
esac
