"""
Zero-shot Octo + OpenVLA evaluation — runs on Mac (CPU/Metal, no CUDA).
Usage:
    python -m baselines.eval_zeroshot_local \
        --data_path ./language_table_episodes.pkl \
        --output_dir ./zeroshot_results \
        --model both   # or: octo / openvla
"""

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
from PIL import Image

STOP_THRESH = 1e-3
NUM_BINS    = 8

ACTION_TOKENS = {
    0: "right", 1: "upright", 2: "up",    3: "upleft",
    4: "left",  5: "downleft", 6: "down", 7: "downright",
}


def discretise(action_2d):
    dx, dy = float(action_2d[0]), float(action_2d[1])
    if abs(dx) < STOP_THRESH and abs(dy) < STOP_THRESH:
        return -1
    return int(round(np.arctan2(dy, dx) / (np.pi / 4))) % NUM_BINS


def load_val_episodes(path, val_frac=0.1):
    with open(path, "rb") as f:
        eps = pickle.load(f)
    random.Random(42).shuffle(eps)
    n = max(1, int(len(eps) * val_frac))
    val = eps[:n]
    windows = sum(len(e["actions"]) for e in val)
    print(f"[data] {len(val)} val episodes, {windows} windows")
    return val


# ─────────────────────────────────────────────────────────────────────────────
# Octo
# ─────────────────────────────────────────────────────────────────────────────

def run_octo(val_eps, out_dir: Path):
    import jax
    import jax.numpy as jnp

    if not hasattr(jax.random, "KeyArray"):
        jax.random.KeyArray = jax.Array
    if not hasattr(jax.random, "PRNGKeyArray"):
        jax.random.PRNGKeyArray = jax.Array

    from octo.model.octo_model import OctoModel

    print("\n[octo] Loading octo-small (CPU) ...")
    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")
    key   = jax.random.PRNGKey(42)
    print("[octo] Loaded. Running zero-shot eval ...")

    correct, total, skipped = 0, 0, 0

    for ep_idx, ep in enumerate(val_eps):
        for t in range(len(ep["actions"])):
            img = Image.fromarray(ep["frames"][t]).resize((256, 256), Image.BILINEAR)
            obs = {"image_primary": np.array(img, dtype=np.float32)[None, None] / 255.0,
                   "timestep_pad_mask": jnp.ones((1, 1), dtype=bool)}
            task     = model.create_tasks(texts=[ep["instruction"]])
            key, rng = jax.random.split(key)
            actions  = model.sample_actions(obs, task, rng=rng)
            pred     = discretise(np.array(actions[0, 0, :2]))
            if pred == -1:
                skipped += 1
                continue
            correct += int(pred == int(ep["actions"][t]))
            total   += 1

        if (ep_idx + 1) % 10 == 0:
            print(f"  [{ep_idx+1}/{len(val_eps)}]  acc={correct/max(total,1)*100:.2f}%")

    acc = correct / total if total > 0 else 0.0
    print(f"\n[octo] Zero-shot accuracy: {acc*100:.2f}%  ({correct}/{total})")
    result = {"model": "octo-small", "mode": "zero_shot", "val_acc": acc,
              "correct": correct, "total": total, "skipped": skipped}
    (out_dir / "octo_zeroshot.json").write_text(json.dumps(result, indent=2))
    print(f"       Saved → {out_dir}/octo_zeroshot.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# OpenVLA  (float16, MPS on Apple Silicon or CPU fallback — no bitsandbytes)
# ─────────────────────────────────────────────────────────────────────────────

def run_openvla(val_eps, out_dir: Path, batch_size: int = 4):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    # Apple Silicon MPS → falls back to CPU if unavailable
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n[openvla] Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("\n[openvla] Using CPU (no MPS/CUDA)")

    print("[openvla] Loading openvla-7b in float16 (no quantisation) ...")
    print("          ~14 GB RAM required — may take a few minutes to load ...")
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print("[openvla] Loaded.")

    action_token_ids = torch.tensor(
        [processor.tokenizer.encode(t, add_special_tokens=False)[0]
         for t in ACTION_TOKENS.values()],
        device=device,
    )

    correct, total = 0, 0
    buf_f, buf_i, buf_b = [], [], []

    def flush():
        nonlocal correct, total
        if not buf_f:
            return
        prompts = [f"In: What action should the robot take to {i}?\nOut:" for i in buf_i]
        inputs  = processor(text=prompts, images=buf_f, return_tensors="pt",
                            padding="longest", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
        preds = logits[:, action_token_ids].argmax(dim=-1).cpu().tolist()
        for pred, gt in zip(preds, buf_b):
            correct += int(pred == gt)
            total   += 1
        buf_f.clear(); buf_i.clear(); buf_b.clear()

    for ep_idx, ep in enumerate(val_eps):
        for t in range(len(ep["actions"])):
            frame = Image.fromarray(ep["frames"][t]).resize((224, 224), Image.BILINEAR)
            buf_f.append(frame)
            buf_i.append(ep["instruction"])
            buf_b.append(int(ep["actions"][t]))
            if len(buf_f) >= batch_size:
                flush()
        if (ep_idx + 1) % 10 == 0:
            flush()
            print(f"  [{ep_idx+1}/{len(val_eps)}]  acc={correct/max(total,1)*100:.2f}%")

    flush()
    acc = correct / total if total > 0 else 0.0
    print(f"\n[openvla] Zero-shot accuracy: {acc*100:.2f}%  ({correct}/{total})")
    result = {"model": "openvla-7b", "mode": "zero_shot", "val_acc": acc,
              "correct": correct, "total": total}
    (out_dir / "openvla_zeroshot.json").write_text(json.dumps(result, indent=2))
    print(f"          Saved → {out_dir}/openvla_zeroshot.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",  required=True)
    p.add_argument("--output_dir", default="./zeroshot_results")
    p.add_argument("--model",      default="both", choices=["octo", "openvla", "both"])
    p.add_argument("--val_frac",   type=float, default=0.1)
    p.add_argument("--batch_size", type=int,   default=4)
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_eps = load_val_episodes(args.data_path, args.val_frac)

    results = {}
    if args.model in ("octo", "both"):
        results["octo"] = run_octo(val_eps, out_dir)

    if args.model in ("openvla", "both"):
        results["openvla"] = run_openvla(val_eps, out_dir, args.batch_size)

    print("\n=== Summary ===")
    print(f"  Random baseline : 12.50%")
    for name, r in results.items():
        print(f"  {name:10s}      : {r['val_acc']*100:.2f}%")


if __name__ == "__main__":
    main()
