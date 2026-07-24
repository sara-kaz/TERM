"""
Octo Fine-Tuning for Language-Table (8-bin action classification)
=================================================================
Loads the pretrained octo-small checkpoint, freezes the transformer backbone,
and adds a trainable linear classification head over 8 directional bins.

Evaluation metric: validation action-classification accuracy (same as TERM).
Random baseline: 12.5% (uniform over 8 bins).

Usage (single seed):
    python -m baselines.finetune_octo_lt \\
        --data_path /data/language_table_episodes.pkl \\
        --output_dir /checkpoints/octo_lt_s42 \\
        --seed 42 --epochs 80

Requirements: octo, jax, flax, optax, tensorflow (for data pipeline),
              numpy, pickle. See nrp/octo_job.yaml for the container image.
"""

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

# ── JAX / Flax / Optax ────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

# JAX 0.4.24 removed jax.random.KeyArray and PRNGKeyArray (replaced by jax.Array).
# Patch before octo is imported — octo/utils/typing.py line 5 still references
# the old names, and the NVIDIA JAX 24.04 image ships JAX 0.4.26+.
if not hasattr(jax.random, "KeyArray"):
    jax.random.KeyArray = jax.Array
if not hasattr(jax.random, "PRNGKeyArray"):
    jax.random.PRNGKeyArray = jax.Array

# ── Octo ──────────────────────────────────────────────────────────────────────
from octo.model.octo_model import OctoModel

def load_episodes(path: str):
    """Load episodes pkl without importing trajectory_dataset (which needs torch)."""
    import pickle
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# Classification head (Flax)
# ─────────────────────────────────────────────────────────────────────────────

class DiscreteActionHead(nn.Module):
    num_actions: int = 8

    @nn.compact
    def __call__(self, x, training: bool = False):
        x = nn.LayerNorm()(x)
        x = nn.Dense(256)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=0.1)(x, deterministic=not training)
        return nn.Dense(self.num_actions)(x)


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE = 256  # octo-small trained on 256×256; patch grid must match
NUM_FRAMES = 3
HISTORY_LEN = 4


def _preprocess_image(frame: np.ndarray) -> np.ndarray:
    """Resize + centre-crop to 224×224, normalise to [0,1]."""
    from PIL import Image
    img = Image.fromarray(frame).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def build_window_refs(episodes: list, seed: int) -> list:
    """Return (ep_idx, step_t) index pairs — NO frame data held in RAM.

    build_windows() used to pre-compute all float32 frame stacks (~2.25 MB each),
    which means 3000 eps × 100 steps × 2.25 MB ≈ 600 GB before training starts.
    Here we only store two integers per window; frames are loaded on demand.
    """
    rng = random.Random(seed)
    refs = []
    for ep_idx, ep in enumerate(episodes):
        T = len(ep["actions"])
        for t in range(NUM_FRAMES - 1, T):
            refs.append((ep_idx, t))
    rng.shuffle(refs)
    return refs


def make_batch_from_refs(episodes: list, batch_refs: list):
    """Load and preprocess one batch worth of frames from raw episode arrays."""
    frames_list, instrs, acts = [], [], []
    for ep_idx, t in batch_refs:
        ep   = episodes[ep_idx]
        idxs = [max(0, t - (NUM_FRAMES - 1 - i)) for i in range(NUM_FRAMES)]
        frame_stack = np.stack([_preprocess_image(ep["frames"][i]) for i in idxs])
        frames_list.append(frame_stack)
        instrs.append(ep["instruction"])
        acts.append(int(ep["actions"][t]))
    return np.stack(frames_list), instrs, np.array(acts, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",  required=True,  help="Path to language_table_episodes.pkl")
    p.add_argument("--output_dir", required=True,  help="Directory for checkpoints and results")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--epochs",     type=int, default=80)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--val_frac",   type=float, default=0.1)
    p.add_argument("--patience",   type=int, default=25,   help="Early stopping patience")
    p.add_argument("--num_actions",type=int, default=8)
    p.add_argument("--octo_ckpt",  default="hf://rail-berkeley/octo-small")
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Reproducibility ───────────────────────────────────────────────────────
    np.random.seed(args.seed)
    random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"[data] Loading episodes from {args.data_path}")
    all_episodes = load_episodes(args.data_path)
    random.Random(42).shuffle(all_episodes)          # fixed shuffle for split reproducibility
    n_val  = max(1, int(len(all_episodes) * args.val_frac))
    val_ep = all_episodes[:n_val]
    trn_ep = all_episodes[n_val:]
    print(f"[data] {len(trn_ep)} train / {len(val_ep)} val episodes")

    trn_refs = build_window_refs(trn_ep, seed=args.seed)
    val_refs  = build_window_refs(val_ep, seed=0)
    print(f"[data] {len(trn_refs)} train / {len(val_refs)} val windows (lazy-loaded per batch)")

    # ── Load Octo backbone ────────────────────────────────────────────────────
    print(f"[octo] Loading pretrained model from {args.octo_ckpt} …")
    octo_model = OctoModel.load_pretrained(args.octo_ckpt)
    print("[octo] Backbone loaded.")

    # ── Build classification head ─────────────────────────────────────────────
    head = DiscreteActionHead(num_actions=args.num_actions)

    # Dummy forward to get feature dim and initialise head params
    dummy_frames  = np.zeros((1, NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    dummy_instrs  = ["push the star near the moon"]
    # Octo expects (batch, timestep, H, W, C) — keep the time dim with [-1:]
    obs_batch = {"image_primary": dummy_frames[:, -1:]}
    task_batch = octo_model.create_tasks(texts=dummy_instrs)
    dummy_out  = octo_model.run_transformer(obs_batch, task_batch, train=False)
    # Extract CLS / readout token from Octo transformer output
    dummy_feat = dummy_out["transformer_outputs"]["readout_action"][:, 0]   # (B, D)

    key, init_key = jax.random.split(key)
    head_vars = head.init(init_key, dummy_feat, training=False)
    print(f"[head] Classification head initialised. Feature dim: {dummy_feat.shape[-1]}")

    # ── Optimiser (head-only) ─────────────────────────────────────────────────
    tx = optax.adamw(learning_rate=args.lr, weight_decay=1e-4)
    opt_state = tx.init(head_vars["params"])

    # ── Training step (jit-compiled) ──────────────────────────────────────────
    @jax.jit
    def train_step(head_params, opt_state, feats, labels):
        def loss_fn(params):
            logits = head.apply({"params": params}, feats, training=True)
            one_hot = jax.nn.one_hot(labels, args.num_actions)
            loss = -jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1).mean()
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(head_params)
        updates, new_opt_state = tx.update(grads, opt_state, head_params)
        new_params = optax.apply_updates(head_params, updates)
        preds = jnp.argmax(logits, axis=-1)
        acc   = (preds == labels).mean()
        return new_params, new_opt_state, loss, acc

    # ── Helper: extract Octo features for a batch ─────────────────────────────
    def extract_features(frames, instrs):
        # Keep time dim: (B, 1, H, W, C) — Octo requires (batch, timestep, H, W, C)
        obs   = {"image_primary": frames[:, -1:]}
        tasks = octo_model.create_tasks(texts=instrs)
        out   = octo_model.run_transformer(obs, tasks, train=False)
        tr = out["transformer_outputs"]
        if "readout_action" not in tr:
            raise KeyError(f"readout_action not in transformer_outputs. Keys: {list(tr.keys())}")
        return np.array(tr["readout_action"][:, 0])

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    patience_ctr = 0
    results = {"train_acc": [], "val_acc": [], "seed": args.seed}
    head_params = head_vars["params"]

    print(f"\n[train] Starting: {args.epochs} epochs, batch {args.batch_size}, lr {args.lr}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        random.shuffle(trn_refs)
        trn_losses, trn_accs = [], []
        for s in range(0, len(trn_refs), args.batch_size):
            batch_refs = trn_refs[s: s + args.batch_size]
            if len(batch_refs) < 2:
                continue
            frames, instrs, acts = make_batch_from_refs(trn_ep, batch_refs)
            feats = extract_features(frames, instrs)
            feats_jax = jnp.array(feats)
            acts_jax  = jnp.array(acts)
            head_params, opt_state, loss, acc = train_step(
                head_params, opt_state, feats_jax, acts_jax)
            trn_losses.append(float(loss))
            trn_accs.append(float(acc))

        # ── Validate ──────────────────────────────────────────────────────────
        val_preds, val_labels = [], []
        for s in range(0, len(val_refs), args.batch_size):
            batch_refs = val_refs[s: s + args.batch_size]
            frames, instrs, acts = make_batch_from_refs(val_ep, batch_refs)
            feats  = extract_features(frames, instrs)
            logits = head.apply({"params": head_params}, jnp.array(feats), training=False)
            val_preds.extend(np.array(jnp.argmax(logits, -1)).tolist())
            val_labels.extend(acts.tolist())

        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        trn_acc = np.mean(trn_accs)
        elapsed = time.time() - t0

        results["train_acc"].append(float(trn_acc))
        results["val_acc"].append(float(val_acc))

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"trn_acc={trn_acc:.4f}  val_acc={val_acc:.4f}  "
              f"loss={np.mean(trn_losses):.4f}  [{elapsed:.1f}s]")

        # ── Early stopping + checkpoint ────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            import flax.serialization as serialization
            ckpt_bytes = serialization.to_bytes(head_params)
            (out / "best_head.msgpack").write_bytes(ckpt_bytes)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"[early stop] No improvement for {args.patience} epochs. Stopping.")
                break

    # ── Save results ──────────────────────────────────────────────────────────
    results["best_val_acc"] = float(best_val_acc)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] Seed {args.seed} | Best val acc: {best_val_acc*100:.2f}%")
    print(f"[done] Results saved to {out}/results.json")


if __name__ == "__main__":
    main()
