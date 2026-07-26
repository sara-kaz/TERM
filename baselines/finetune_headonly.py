"""
Frozen backbone + MLP head fine-tuning for Octo and OpenVLA.
Phase 1: extract embeddings with frozen backbone (no grad).
Phase 2: train a 3-layer MLP on cached embeddings.
~20-30 min total on one GPU.

Usage:
    python -m baselines.finetune_headonly --model octo    --data_path /data/lt.pkl --output_dir /results
    python -m baselines.finetune_headonly --model openvla --data_path /data/lt.pkl --output_dir /results
"""

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from PIL import Image

NUM_BINS     = 8
IMG_OCTO     = 256
IMG_VLA      = 224




# ── Data ──────────────────────────────────────────────────────────────────────

def split_episodes(path, val_frac=0.1, seed=42):
    with open(path, "rb") as f:
        eps = pickle.load(f)
    random.Random(seed).shuffle(eps)
    n = max(1, int(len(eps) * val_frac))
    return eps[n:], eps[:n]   # train, val


# ── Octo embedding extraction ─────────────────────────────────────────────────

def extract_octo(episodes, ckpt="hf://rail-berkeley/octo-small"):
    import jax
    import jax.numpy as jnp

    if not hasattr(jax.random, "KeyArray"):
        jax.random.KeyArray = jax.Array
    if not hasattr(jax.random, "PRNGKeyArray"):
        jax.random.PRNGKeyArray = jax.Array

    from octo.model.octo_model import OctoModel

    print(f"[octo] Loading {ckpt} ...")
    model = OctoModel.load_pretrained(ckpt)
    print(f"[octo] Loaded. Extracting embeddings from {len(episodes)} episodes ...")

    embs, labels = [], []
    for ep_idx, ep in enumerate(episodes):
        for t in range(len(ep["actions"])):
            img   = Image.fromarray(ep["frames"][t]).resize((IMG_OCTO, IMG_OCTO), Image.BILINEAR)
            frame = np.array(img, dtype=np.float32)[None, None] / 255.0   # (1,1,H,W,3)
            obs   = {"image_primary": frame,
                     "timestep_pad_mask": jnp.ones((1, 1), dtype=bool)}
            task  = model.create_tasks(texts=[ep["instruction"]])
            pad_mask = jnp.ones((1, 1), dtype=bool)

            out = model.run_transformer(obs, task, pad_mask, train=False)
            # readout_action.tokens: (batch, window, num_tokens, d_model)
            tok = np.array(out["readout_action"].tokens[0, 0])   # (num_tokens, d_model)
            embs.append(tok.flatten())
            labels.append(int(ep["actions"][t]))

        if (ep_idx + 1) % 50 == 0:
            print(f"  [{ep_idx+1}/{len(episodes)}]  {len(embs)} steps cached")

    return np.stack(embs).astype(np.float32), np.array(labels, dtype=np.int64)


# ── OpenVLA embedding extraction ──────────────────────────────────────────────

def extract_openvla(episodes, model_id="openvla/openvla-7b", batch_size=8):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[openvla] Loading {model_id} (4-bit) on {device} ...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, quantization_config=bnb, device_map={"": 0}, trust_remote_code=True,
    )
    model.eval()
    print(f"[openvla] Loaded. Extracting embeddings from {len(episodes)} episodes ...")

    all_frames, all_instrs, all_bins = [], [], []
    for ep in episodes:
        for t in range(len(ep["actions"])):
            frame = Image.fromarray(ep["frames"][t]).resize((IMG_VLA, IMG_VLA), Image.BILINEAR)
            all_frames.append(frame)
            all_instrs.append(ep["instruction"])
            all_bins.append(int(ep["actions"][t]))

    n       = len(all_frames)
    prompts = [f"In: What action should the robot take to {i}?\nOut:" for i in all_instrs]
    embs    = []

    for start in range(0, n, batch_size):
        end    = min(start + batch_size, n)
        inputs = processor(
            text=prompts[start:end], images=all_frames[start:end],
            return_tensors="pt", padding="longest", truncation=True, max_length=64,
        ).to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, return_dict=True)
            # last hidden state at the final input token = "action slot"
            h = out.hidden_states[-1][:, -1, :].float().cpu().numpy()   # (B, 4096)
        embs.append(h)

        if (start // batch_size + 1) % 25 == 0:
            print(f"  [{end}/{n}]")

    return np.concatenate(embs, axis=0).astype(np.float32), np.array(all_bins, dtype=np.int64)


# ── MLP training (sklearn — works in JAX and PyTorch images alike) ────────────

def train_mlp(tr_emb, tr_lbl, vl_emb, vl_lbl, d_in, num_epochs=200, seed=42, **_):
    print(f"[mlp] Fitting MLP ({d_in}→256→64→8), {num_epochs} max iters, seed={seed} ...")
    scaler   = StandardScaler()
    tr_scaled = scaler.fit_transform(tr_emb)
    vl_scaled = scaler.transform(vl_emb)

    clf = MLPClassifier(
        hidden_layer_sizes=(256, 64),
        activation="relu",
        solver="adam",
        max_iter=num_epochs,
        random_state=seed,
        verbose=False,
    )
    clf.fit(tr_scaled, tr_lbl)
    acc = clf.score(vl_scaled, vl_lbl)
    print(f"[mlp] Val acc: {acc*100:.2f}%  (iters run: {clf.n_iter_})")
    return acc


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True, choices=["octo", "openvla"])
    p.add_argument("--data_path",   required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--val_frac",    type=float, default=0.1)
    p.add_argument("--num_epochs",  type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--seed",        type=int,   default=42,
                   help="Random seed for train/val split and MLP init (use 42, 123, 456 to match TERM)")
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_eps, val_eps = split_episodes(args.data_path, args.val_frac, seed=args.seed)
    print(f"[data] seed={args.seed}  train={len(train_eps)} eps, val={len(val_eps)} eps")

    if args.model == "octo":
        tr_emb, tr_lbl = extract_octo(train_eps)
        vl_emb, vl_lbl = extract_octo(val_eps)
        model_name = "octo-small"
    else:
        tr_emb, tr_lbl = extract_openvla(train_eps, batch_size=args.batch_size)
        vl_emb, vl_lbl = extract_openvla(val_eps,   batch_size=args.batch_size)
        model_name = "openvla-7b"

    d_in = tr_emb.shape[1]
    print(f"\n[emb] d_in={d_in}  train_steps={len(tr_lbl)}  val_steps={len(vl_lbl)}")

    best_acc = train_mlp(tr_emb, tr_lbl, vl_emb, vl_lbl, d_in, args.num_epochs, seed=args.seed)

    result = {
        "model": model_name, "mode": "headonly_ft",
        "seed": args.seed,
        "val_acc": best_acc, "d_in": d_in,
        "train_steps": int(len(tr_lbl)), "val_steps": int(len(vl_lbl)),
        "num_epochs": args.num_epochs,
    }
    fname = f"results_s{args.seed}.json"
    (out / fname).write_text(json.dumps(result, indent=2))
    print(f"\n[done] {model_name} head-only  seed={args.seed}  val_acc={best_acc*100:.2f}%")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
