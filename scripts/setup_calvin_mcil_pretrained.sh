#!/usr/bin/env bash
# Download official CALVIN MCIL D→D pretrained weights + language embeddings.
set -euo pipefail

CALVIN_DATA="${CALVIN_DATA:-$HOME/calvin_task_D/task_D_D}"
CALVIN_ROOT="${CALVIN_ROOT:-$HOME/work/calvin}"
WEIGHTS_DIR="${MCIL_WEIGHTS_DIR:-$HOME/work/RLConditionedVLA/checkpoints/calvin_mcil_pretrained/D_D_static_rgb_baseline}"
ZIP_URL="${MCIL_ZIP_URL:-http://calvin.cs.uni-freiburg.de/model_weights/D_D_static_rgb_baseline.zip}"

mkdir -p "$(dirname "$WEIGHTS_DIR")"

if [[ -f "$WEIGHTS_DIR/.hydra/config.yaml" ]]; then
  echo "MCIL weights already at $WEIGHTS_DIR"
else
  echo "Downloading MCIL pretrained weights..."
  TMP="$(mktemp -d)"
  wget -O "$TMP/mcil.zip" "$ZIP_URL"
  unzip -q "$TMP/mcil.zip" -d "$(dirname "$WEIGHTS_DIR")"
  rm -rf "$TMP"
  # Zip may extract to D_D_static_rgb_baseline/ or similar
  if [[ ! -d "$WEIGHTS_DIR" ]]; then
    FOUND="$(find "$(dirname "$WEIGHTS_DIR")" -maxdepth 2 -name config.yaml -path '*/.hydra/*' 2>/dev/null | head -1)"
    if [[ -n "$FOUND" ]]; then
      WEIGHTS_DIR="$(dirname "$(dirname "$FOUND")")"
      echo "Detected train_folder: $WEIGHTS_DIR"
    fi
  fi
fi

# Language embeddings (required for MCIL eval)
if [[ -f "$CALVIN_DATA/validation/lang_embeddings/embeddings.npy" ]]; then
  echo "Lang embeddings OK"
else
  echo "Downloading language embeddings for split D..."
  if [[ -f "$CALVIN_ROOT/dataset/download_lang_embeddings.sh" ]]; then
    cd "$CALVIN_ROOT/dataset"
    bash download_lang_embeddings.sh D
  else
    echo "WARN: run manually: cd \$CALVIN_ROOT/dataset && sh download_lang_embeddings.sh D"
  fi
fi

CKPT="$(find "$WEIGHTS_DIR" -name '*.ckpt' 2>/dev/null | head -1)"
echo ""
echo "Setup complete."
echo "  train_folder=$WEIGHTS_DIR"
echo "  checkpoint=${CKPT:-<run: find $WEIGHTS_DIR -name '*.ckpt'>}"
echo "  dataset=$CALVIN_DATA"
