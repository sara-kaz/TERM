#!/usr/bin/env bash
# Download CALVIN D→D (task_D_D.zip), optionally extract when the archive is complete.
#
# The zip is large (~170GB compressed); use tmux/screen and free disk for zip + unpacked data.
#
# Usage (use a REAL directory you own — NOT the literal "/path/..." docs text):
#   DOWNLOAD_ONLY=1 bash scripts/download_calvin_task_d_d.sh "$HOME/calvin_task_D"
#   bash scripts/download_calvin_task_d_d.sh "$HOME/calvin_task_D"
#
# Env:
#   DOWNLOAD_URL — override full ZIP URL (optional mirror)
#   DOWNLOAD_ONLY=1 — fetch only; skip unzip
#
# Official README uses HTTP (port 80), not HTTPS — if :443 replies "connection refused",
# rely on defaults below or pass DOWNLOAD_URL=http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip
set -euo pipefail

DEST="${1:?Pass a writable directory, e.g. \$HOME/calvin_task_D (not /path/with/enough/space)}"

# Catch copied documentation placeholder literally
if [[ "$DEST" == /path/* ]] || [[ "$DEST" == */with/enough/space ]] || [[ "$DEST" == */with/enough/space/ ]]; then
  echo "[calvin] \"$DEST\" is a documentation placeholder, not a real folder."
  echo "Use something under your home directory with enough disk space, e.g.:"
  echo "  DOWNLOAD_ONLY=1 bash \"$0\" \"\$HOME/calvin_task_D\""
  exit 1
fi

if [[ "$DEST" == "~" ]] || [[ "$DEST" =~ ^~/ ]]; then
  echo "[calvin] Quote paths with \$HOME instead of tilde (~ is not expanded in variables)."
  echo "  Example: \"$0\" \"\$HOME/calvin_task_D\""
  exit 1
fi

if ! mkdir -p "$DEST"; then
  echo "[calvin] Cannot mkdir: $DEST"
  echo "Choose a writable path with free space (~200GB+ for zip + extract), e.g. \$HOME/calvin_task_D"
  exit 1
fi
# Official hosting (~166 GB). See github.com/mees/calvin — dataset/README.md
URL_HTTP="${URL_HTTP:-http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip}"
URL_HTTPS="${URL_HTTPS:-https://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip}"
ZIP_NAME="task_D_D.zip"
cd "$DEST"

wget_zip() {
  wget --continue --tries=0 --retry-connrefused --waitretry=5 -O "$ZIP_NAME" "$1"
}

if [[ -n "${DOWNLOAD_URL:-}" ]]; then
  echo "[calvin] Downloading from \$DOWNLOAD_URL → $DEST/$ZIP_NAME"
  echo "           $DOWNLOAD_URL"
  wget_zip "$DOWNLOAD_URL"
elif [[ ! -f "$ZIP_NAME" ]]; then
  echo "[calvin] Trying HTTP first (official) → $DEST/$ZIP_NAME"
  if ! wget_zip "$URL_HTTP"; then
    echo "[calvin] HTTP failed; retrying HTTPS..."
    wget_zip "$URL_HTTPS"
  fi
else
  echo "[calvin] Continuing partial download: $DEST/$ZIP_NAME"
  if ! wget_zip "$URL_HTTP"; then
    echo "[calvin] HTTP resume failed; trying HTTPS..."
    wget_zip "$URL_HTTPS"
  fi
fi

if [[ "${DOWNLOAD_ONLY:-0}" == "1" ]]; then
  echo "[calvin] DOWNLOAD_ONLY=1 → skip unzip. When done downloading, run:"
  echo "  DOWNLOAD_ONLY=0 bash scripts/download_calvin_task_d_d.sh \"$DEST\""
  exit 0
fi

echo "[calvin] Verifying ZIP integrity (quick test)..."
if ! unzip -tq "$ZIP_NAME" 2>/dev/null; then
  echo "[calvin] ZIP not complete or corrupt yet. Resume download with:"
  echo "  DOWNLOAD_ONLY=1 bash \"$0\" \"$DEST\""
  exit 1
fi

echo "[calvin] Extracting (requires additional free disk; can take tens of minutes)..."
unzip -q -o "$ZIP_NAME"

TASK_DIR="$DEST/task_D_D"
if [[ -d "$TASK_DIR" ]]; then
  echo "[calvin] Ready. Train with:"
  echo "  python scripts/run_calvin_comparisons.py --calvin_path \"$TASK_DIR\" ..."
else
  echo "[calvin] Extract done but $TASK_DIR not found — inspect $DEST contents."
fi
