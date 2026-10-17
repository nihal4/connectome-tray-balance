#!/usr/bin/env bash
# One-time setup of an always-on Ubuntu training PC (tested target: Ubuntu 24.04).
#   bash <(curl -fsSL https://raw.githubusercontent.com/nihal4/connectome-tray-balance/main/scripts/desktop/setup.sh)
# or, from a clone:  bash scripts/desktop/setup.sh
#
# Creates ~/flybrain/{repo,.venv,runs,logs}, installs CPU PyTorch + MuJoCo, downloads the
# connectome, builds the brains, checks everything, and installs:
#   flybrain-train   training service (resumes after reboot / power cut)
#   flybrain-watch   progress videos every 100k steps + dashboard page
#   flybrain-web     dashboard at http://<this-pc>:8080
#   fly              command to control it all (fly help)
set -euo pipefail
ROOT="$HOME/flybrain"
REPO="$ROOT/repo"
PY="$ROOT/.venv/bin/python"

echo "== system packages (asks for your password once)"
sudo apt-get update -qq
sudo apt-get install -y -qq git python3-venv python3-dev libegl1 libgl1 libosmesa6 ffmpeg tmux lm-sensors zip >/dev/null

echo "== code"
mkdir -p "$ROOT/runs" "$ROOT/logs"
if [ -d "$REPO/.git" ]; then git -C "$REPO" pull -q; else git clone -q https://github.com/nihal4/connectome-tray-balance "$REPO"; fi

echo "== python environment (CPU PyTorch: smaller, no GPU needed)"
[ -x "$PY" ] || python3 -m venv "$ROOT/.venv"
"$PY" -m pip install -q -U pip
"$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
"$PY" -m pip install -q -r "$REPO/requirements.txt"

echo "== connectome + brains"
cd "$REPO"
"$PY" scripts/download_connectome.py --out data/raw
[ -f data/flybrain_eye160_N4000.npz ] || "$PY" scripts/build_flybrain.py --eye compound --facets 160 --n 4000 --hops 5 >/dev/null
[ -f data/flybrain_cam16_N2500.npz ] || "$PY" scripts/build_flybrain.py >/dev/null
[ -f data/subset_flywire783_N1000.npz ] || "$PY" scripts/build_subset.py --n 1000 >/dev/null
ls data/*.npz

echo "== quick check"
"$PY" scripts/check_mac.py --env drone_reach --subset data/flybrain_eye160_N4000.npz | tail -12

echo "== services"
mkdir -p "$HOME/.config/systemd/user" "$HOME/.local/bin"
ln -sf "$REPO/scripts/desktop/fly" "$HOME/.local/bin/fly"
chmod +x "$REPO/scripts/desktop/fly" "$REPO/scripts/desktop/run_job.sh"
cat > "$HOME/.config/systemd/user/flybrain-train.service" <<UNIT
[Unit]
Description=Fly-brain training (resumes from the last checkpoint)
After=network-online.target
[Service]
ExecStart=$REPO/scripts/desktop/run_job.sh
Restart=on-failure
RestartSec=60
Nice=5
[Install]
WantedBy=default.target
UNIT
cat > "$HOME/.config/systemd/user/flybrain-watch.service" <<UNIT
[Unit]
Description=Fly-brain progress videos and dashboard
[Service]
ExecStart=$PY $REPO/scripts/desktop/watch.py --root $ROOT
Restart=always
RestartSec=60
Nice=19
Environment=MUJOCO_GL=egl
[Install]
WantedBy=default.target
UNIT
cat > "$HOME/.config/systemd/user/flybrain-web.service" <<UNIT
[Unit]
Description=Fly-brain dashboard on port 8080
[Service]
ExecStart=$PY -m http.server 8080 --directory $ROOT
Restart=always
[Install]
WantedBy=default.target
UNIT
[ -f "$ROOT/job.env" ] || "$REPO/scripts/desktop/fly" job drone_reach >/dev/null
sudo loginctl enable-linger "$USER"          # user services start at boot, without logging in
systemctl --user daemon-reload
systemctl --user enable --now flybrain-watch flybrain-web
systemctl --user enable flybrain-train       # starts at boot from now on; 'fly start' starts it now
if sudo ufw status 2>/dev/null | grep -q "Status: active"; then sudo ufw allow 8080/tcp >/dev/null; fi

IP=$(hostname -I | awk '{print $1}')
echo
echo "Done. Open a new terminal (or run: source ~/.profile), then:"
echo "   fly job       # show what will be trained"
echo "   fly start     # start (it also starts by itself after every reboot)"
echo "   fly status    # progress"
echo "Dashboard: http://$IP:8080"
