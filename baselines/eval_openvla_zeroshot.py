"""
Zero-shot OpenVLA evaluation on Language-Table (8-bin action accuracy).
No fine-tuning — loads pretrained openvla-7b, restricts the next-token
logits to the 8 directional action words, and takes the argmax.
"""

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

ACTION_TOKENS = {
    0: "right",
    1: "upright",
    2: "up",
    3: "upleft",
    4: "left",
    5: "downleft",
    6: "down",
    7: "downright",
}
IMG_SIZE = 224


def load_episodes(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",  required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--val_frac",   type=float, default=0.1)
    p.add_argument("--model_id",   default="openvla/openvla-7b")
    p.add_argument("--batch_size", type=int,   default=8)
    return p.parse_args()


def main():
    args   = parse_args()
    out    = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[data] Loading {args.data_path}")
    all_eps = load_episodes(args.data_path)
    random.Random(42).shuffle(all_eps)
    n_val   = max(1, int(len(all_eps) * args.val_frac))
    val_eps = all_eps[:n_val]
    print(f"[data] {len(val_eps)} val episodes")

    print(f"[model] Loading {args.model_id} (4-bit QLoRA) ...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    bnb_cfg   = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        quantization_config=bnb_cfg,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()
    print("[model] Loaded.")

    # Cache the token ids for each action word once
    action_token_ids = torch.tensor(
        [processor.tokenizer.encode(t, add_special_tokens=False)[0]
         for t in ACTION_TOKENS.values()],
        device=device,
    )

    correct, total = 0, 0
    buf_frames, buf_instrs, buf_bins = [], [], []

    def flush():
        nonlocal correct, total
        if not buf_frames:
            return
        prompts = [
            f"In: What action should the robot take to {instr}?\nOut:"
            for instr in buf_instrs
        ]
        inputs = processor(
            text=prompts,
            images=buf_frames,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=64,
        ).to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]   # (B, vocab)
        action_logits = logits[:, action_token_ids]      # (B, 8)
        preds = action_logits.argmax(dim=-1).cpu().tolist()
        for pred, gt in zip(preds, buf_bins):
            correct += int(pred == gt)
            total   += 1
        buf_frames.clear()
        buf_instrs.clear()
        buf_bins.clear()

    for ep_idx, ep in enumerate(val_eps):
        for t in range(len(ep["actions"])):
            frame = Image.fromarray(ep["frames"][t]).resize(
                (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            buf_frames.append(frame)
            buf_instrs.append(ep["instruction"])
            buf_bins.append(int(ep["actions"][t]))
            if len(buf_frames) >= args.batch_size:
                flush()

        if (ep_idx + 1) % 25 == 0:
            flush()
            acc = correct / max(total, 1) * 100
            print(f"  [{ep_idx+1}/{len(val_eps)}]  acc={acc:.2f}%  ({correct}/{total})")

    flush()

    acc = correct / total if total > 0 else 0.0
    results = {
        "model":   "openvla-7b",
        "mode":    "zero_shot",
        "val_acc": acc,
        "correct": correct,
        "total":   total,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] Zero-shot OpenVLA accuracy: {acc*100:.2f}%  ({correct}/{total})")


if __name__ == "__main__":
    main()
