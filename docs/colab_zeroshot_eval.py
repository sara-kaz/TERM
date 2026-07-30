# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot Octo + OpenVLA evaluation on Language-Table
# Run this in Google Colab (GPU runtime: T4 or A100)
#
# BEFORE RUNNING:
#   1. Upload language_table_episodes.pkl to your Google Drive
#   2. Set DRIVE_PKL_PATH below to where you put it
#   3. Runtime → Change runtime type → GPU (T4 free, A100 Colab Pro)
# ─────────────────────────────────────────────────────────────────────────────

# ── CELL 1: Mount Drive ───────────────────────────────────────────────────────
from google.colab import drive
drive.mount("/content/drive")

DRIVE_PKL_PATH = "/content/drive/MyDrive/language_table_episodes.pkl"  # ← change if needed
RESULTS_DIR    = "/content/drive/MyDrive/zeroshot_results"

import os
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── CELL 2: Install dependencies (run once, then restart runtime if prompted) ─
# Octo needs JAX (already in Colab), OpenVLA needs transformers + bitsandbytes
# Install everything in one shot — no restart needed if you run this first.

import subprocess
cmds = [
    "pip install -q 'tensorflow-datasets>=4.5.0,<4.9.0' 'tensorflow-metadata<1.14.0'",
    "pip install -q 'tensorflow-probability>=0.22.0,<0.24.0'",
    "pip install -q distrax einops ml-collections absl-py huggingface_hub "
        "'transformers>=4.41.0,<4.45.0' sentencepiece",
    "pip install -q 'git+https://github.com/kvablack/dlimp.git'",
    "pip install -q 'git+https://github.com/octo-models/octo.git' flax optax chex",
    "pip install -q bitsandbytes 'timm>=0.9.10,<1.0.0' accelerate peft pillow",
]
for cmd in cmds:
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)
print("Done.")

# ── CELL 3: Load data ─────────────────────────────────────────────────────────
import pickle, random
import numpy as np

print(f"Loading {DRIVE_PKL_PATH} ...")
with open(DRIVE_PKL_PATH, "rb") as f:
    all_eps = pickle.load(f)

random.Random(42).shuffle(all_eps)
n_val   = max(1, int(len(all_eps) * 0.1))
val_eps = all_eps[:n_val]
print(f"Val episodes: {len(val_eps)}")
total_windows = sum(len(ep["actions"]) for ep in val_eps)
print(f"Val windows:  {total_windows}")

# ── CELL 4: Zero-shot OCTO ───────────────────────────────────────────────────
import jax
import jax.numpy as jnp
from PIL import Image

if not hasattr(jax.random, "KeyArray"):
    jax.random.KeyArray = jax.Array
if not hasattr(jax.random, "PRNGKeyArray"):
    jax.random.PRNGKeyArray = jax.Array

from octo.model.octo_model import OctoModel

IMG_SIZE_OCTO = 256
STOP_THRESH   = 1e-3

def discretise(action_2d):
    dx, dy = float(action_2d[0]), float(action_2d[1])
    if abs(dx) < STOP_THRESH and abs(dy) < STOP_THRESH:
        return -1
    angle = np.arctan2(dy, dx)
    return int(round(angle / (np.pi / 4))) % 8

def preprocess_octo(frame):
    img = Image.fromarray(frame).resize((IMG_SIZE_OCTO, IMG_SIZE_OCTO), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0

print("Loading octo-small ...")
octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")
key = jax.random.PRNGKey(42)
print("Loaded. Running evaluation ...")

correct_octo, total_octo, skipped_octo = 0, 0, 0

for ep_idx, ep in enumerate(val_eps):
    for t in range(len(ep["actions"])):
        frame    = preprocess_octo(ep["frames"][t])
        obs      = {"image_primary": frame[None, None]}
        task     = octo_model.create_tasks(texts=[ep["instruction"]])
        key, rng = jax.random.split(key)
        actions  = octo_model.sample_actions(obs, task, rng=rng, unnormalize_actions=False)
        act_xy   = np.array(actions[0, 0, :2])
        pred     = discretise(act_xy)
        if pred == -1:
            skipped_octo += 1
            continue
        correct_octo += int(pred == int(ep["actions"][t]))
        total_octo   += 1

    if (ep_idx + 1) % 25 == 0:
        print(f"  [{ep_idx+1}/{len(val_eps)}] acc={correct_octo/max(total_octo,1)*100:.2f}%")

octo_acc = correct_octo / total_octo if total_octo > 0 else 0.0
print(f"\nOcto zero-shot accuracy: {octo_acc*100:.2f}%  ({correct_octo}/{total_octo})")

import json
octo_result = {"model": "octo-small", "mode": "zero_shot",
               "val_acc": octo_acc, "correct": correct_octo, "total": total_octo}
with open(f"{RESULTS_DIR}/octo_zeroshot.json", "w") as f:
    json.dump(octo_result, f, indent=2)
print(f"Saved → {RESULTS_DIR}/octo_zeroshot.json")

# ── CELL 5: Zero-shot OpenVLA ─────────────────────────────────────────────────
# Note: if Colab RAM is tight after Octo, restart runtime, re-run Cells 1+3,
# then jump straight here (OpenVLA does not need JAX/Octo loaded).

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

ACTION_TOKENS = {0:"right",1:"upright",2:"up",3:"upleft",
                 4:"left",5:"downleft",6:"down",7:"downright"}
IMG_SIZE_OPENVLA = 224
BATCH_SIZE       = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading openvla-7b (4-bit) ...")
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.float16,
                          bnb_4bit_use_double_quant=True)
openvla_model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", quantization_config=bnb,
    device_map={"": 0}, trust_remote_code=True)
openvla_model.eval()
print("Loaded.")

action_token_ids = torch.tensor(
    [processor.tokenizer.encode(t, add_special_tokens=False)[0]
     for t in ACTION_TOKENS.values()], device=device)

correct_vla, total_vla = 0, 0
buf_frames, buf_instrs, buf_bins = [], [], []

def flush_vla():
    global correct_vla, total_vla
    if not buf_frames:
        return
    prompts = [f"In: What action should the robot take to {i}?\nOut:"
               for i in buf_instrs]
    inputs = processor(text=prompts, images=buf_frames,
                       return_tensors="pt", padding="longest",
                       truncation=True, max_length=64).to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    with torch.no_grad():
        logits = openvla_model(**inputs).logits[:, -1, :]
    preds = logits[:, action_token_ids].argmax(dim=-1).cpu().tolist()
    for pred, gt in zip(preds, buf_bins):
        correct_vla += int(pred == gt)
        total_vla   += 1
    buf_frames.clear(); buf_instrs.clear(); buf_bins.clear()

for ep_idx, ep in enumerate(val_eps):
    for t in range(len(ep["actions"])):
        frame = Image.fromarray(ep["frames"][t]).resize(
            (IMG_SIZE_OPENVLA, IMG_SIZE_OPENVLA), Image.BILINEAR)
        buf_frames.append(frame)
        buf_instrs.append(ep["instruction"])
        buf_bins.append(int(ep["actions"][t]))
        if len(buf_frames) >= BATCH_SIZE:
            flush_vla()
    if (ep_idx + 1) % 25 == 0:
        flush_vla()
        print(f"  [{ep_idx+1}/{len(val_eps)}] acc={correct_vla/max(total_vla,1)*100:.2f}%")

flush_vla()

vla_acc = correct_vla / total_vla if total_vla > 0 else 0.0
print(f"\nOpenVLA zero-shot accuracy: {vla_acc*100:.2f}%  ({correct_vla}/{total_vla})")

vla_result = {"model": "openvla-7b", "mode": "zero_shot",
              "val_acc": vla_acc, "correct": correct_vla, "total": total_vla}
with open(f"{RESULTS_DIR}/openvla_zeroshot.json", "w") as f:
    json.dump(vla_result, f, indent=2)
print(f"Saved → {RESULTS_DIR}/openvla_zeroshot.json")

# ── CELL 6: Summary ───────────────────────────────────────────────────────────
print("\n=== Zero-shot Results ===")
print(f"Random baseline : 12.50%")
print(f"Octo zero-shot  : {octo_acc*100:.2f}%")
print(f"OpenVLA zero-shot: {vla_acc*100:.2f}%")
print(f"\nResults saved to {RESULTS_DIR}/")
