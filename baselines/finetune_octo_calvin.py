"""
Octo End-to-End Fine-Tuning for CALVIN D→D (14-bin action classification)
=========================================================================
Same protocol as finetune_octo_lt.py but with CALVIN's 14-bin arm-priority
discretisation and the task_D_D dataset.

14-bin scheme (matches VERA CALVIN):
  Bins 0–11 : dominant arm DoF (6 DoF × pos/neg)
  Bins 12–13: gripper open / close (only when arm motion < threshold)

Backbone LR: 1e-5  Head LR: 1e-4  (same as LT baseline)

Usage:
    python -m baselines.finetune_octo_calvin \\
        --data_path /data/calvin_episodes.pkl \\
        --output_dir /results \\
        --seed 42 --epochs 20
"""

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import flax.serialization as serialization
import optax

if not hasattr(jax.random, "KeyArray"):
    jax.random.KeyArray = jax.Array
if not hasattr(jax.random, "PRNGKeyArray"):
    jax.random.PRNGKeyArray = jax.Array

from octo.model.octo_model import OctoModel


# ─────────────────────────────────────────────────────────────────────────────
# Classification head
# ─────────────────────────────────────────────────────────────────────────────

class DiscreteActionHead(nn.Module):
    num_actions: int = 14

    @nn.compact
    def __call__(self, x, training: bool = False):
        x = nn.LayerNorm()(x)
        x = nn.Dense(256)(x)
        x = nn.gelu(x)
        return nn.Dense(self.num_actions)(x)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE   = 256
NUM_FRAMES = 3


def load_episodes(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _preprocess_image(frame: np.ndarray) -> np.ndarray:
    from PIL import Image
    img = Image.fromarray(frame).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def build_window_refs(episodes: list, seed: int) -> list:
    rng = random.Random(seed)
    refs = []
    for ep_idx, ep in enumerate(episodes):
        T = len(ep["actions"])
        for t in range(NUM_FRAMES - 1, T):
            refs.append((ep_idx, t))
    rng.shuffle(refs)
    return refs


def build_task_cache(octo_model, episodes, chunk_size: int = 64):
    _ANCHOR = (
        "move the robotic arm to pick up and relocate "
        "the target object to the destination zone"
    )
    unique_texts = list({ep["instruction"] for ep in episodes})
    print(f"[tasks] Pre-computing embeddings for {len(unique_texts)} unique instructions …")
    cache: dict = {}
    for i in range(0, len(unique_texts), chunk_size):
        chunk = [_ANCHOR] + unique_texts[i : i + chunk_size]
        tasks = octo_model.create_tasks(texts=chunk)
        for j, txt in enumerate(unique_texts[i : i + chunk_size]):
            cache[txt] = jax.tree_util.tree_map(lambda x: x[j + 1 : j + 2], tasks)
    print(f"[tasks] Cache built ({len(cache)} entries).")
    return cache


def make_batch_from_refs(episodes, batch_refs, task_cache):
    frames_list, task_list, acts = [], [], []
    for ep_idx, t in batch_refs:
        ep   = episodes[ep_idx]
        idxs = [max(0, t - (NUM_FRAMES - 1 - i)) for i in range(NUM_FRAMES)]
        frame_stack = np.stack([_preprocess_image(ep["frames"][i]) for i in idxs])
        frames_list.append(frame_stack)
        task_list.append(task_cache[ep["instruction"]])
        acts.append(int(ep["actions"][t]))
    tasks = jax.tree_util.tree_map(
        lambda *arrs: jnp.concatenate(arrs, axis=0), *task_list
    )
    return np.stack(frames_list), tasks, np.array(acts, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(ckpt_dir: Path, all_params, opt_state, state: dict):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "params.msgpack").write_bytes(serialization.to_bytes(all_params))
    with open(ckpt_dir / "opt_state.pkl", "wb") as f:
        pickle.dump(opt_state, f)
    (ckpt_dir / "state.json").write_text(json.dumps(state, indent=2))


def _load_checkpoint(ckpt_dir: Path, all_params_template, opt_state_template):
    state = json.loads((ckpt_dir / "state.json").read_text())
    all_params = serialization.from_bytes(
        all_params_template, (ckpt_dir / "params.msgpack").read_bytes())
    with open(ckpt_dir / "opt_state.pkl", "rb") as f:
        opt_state = pickle.load(f)
    return all_params, opt_state, state


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",    required=True)
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--resume_from",  default=None)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr_backbone",  type=float, default=1e-5)
    p.add_argument("--lr_head",      type=float, default=1e-4)
    p.add_argument("--val_frac",     type=float, default=0.1)
    p.add_argument("--patience",     type=int,   default=10)
    p.add_argument("--num_actions",  type=int,   default=14)
    p.add_argument("--octo_ckpt",    default="hf://rail-berkeley/octo-small")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoint"

    np.random.seed(args.seed)
    random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    print(f"[data] Loading {args.data_path}")
    all_episodes = load_episodes(args.data_path)
    random.Random(42).shuffle(all_episodes)
    n_val  = max(1, int(len(all_episodes) * args.val_frac))
    val_ep = all_episodes[:n_val]
    trn_ep = all_episodes[n_val:]
    print(f"[data] {len(trn_ep)} train / {len(val_ep)} val episodes")

    trn_refs = build_window_refs(trn_ep, seed=args.seed)
    val_refs = build_window_refs(val_ep, seed=0)
    print(f"[data] {len(trn_refs)} train / {len(val_refs)} val windows")

    jax.config.update("jax_disable_jit", True)
    print(f"[octo] Loading {args.octo_ckpt} …")
    octo_model = OctoModel.load_pretrained(args.octo_ckpt)
    print("[octo] Backbone loaded.")
    jax.config.update("jax_disable_jit", False)

    all_ep = trn_ep + val_ep
    task_cache = build_task_cache(octo_model, all_ep)

    head = DiscreteActionHead(num_actions=args.num_actions)
    _sample_task = task_cache[next(iter(task_cache))]
    dummy_obs  = {"image_primary": np.zeros((1, 1, IMG_SIZE, IMG_SIZE, 3), np.float32)}
    dummy_msk  = jnp.ones((1, 1), dtype=bool)
    dummy_out  = octo_model.run_transformer(dummy_obs, _sample_task, dummy_msk, train=False)
    dummy_feat = dummy_out["readout_action"].tokens[:, 0]

    key, init_key = jax.random.split(key)
    head_vars = head.init(init_key, dummy_feat, training=False)
    print(f"[head] Initialised. Feature dim: {dummy_feat.shape[-1]}")

    backbone_params = flax.core.unfreeze(octo_model.params)
    all_params = {"backbone": backbone_params, "head": head_vars["params"]}

    tx = optax.multi_transform(
        transforms={
            "backbone": optax.adamw(learning_rate=args.lr_backbone, weight_decay=1e-5),
            "head":     optax.adamw(learning_rate=args.lr_head,     weight_decay=1e-4),
        },
        param_labels={"backbone": "backbone", "head": "head"},
    )
    opt_state = tx.init(all_params)

    start_epoch  = 1
    best_val_acc = 0.0
    patience_ctr = 0
    results      = {"train_acc": [], "val_acc": [], "seed": args.seed}

    all_params_template = all_params
    opt_state_template  = opt_state

    resume_path = Path(args.resume_from) if args.resume_from else (
        ckpt_dir if (ckpt_dir / "state.json").exists() else None
    )
    if resume_path and (resume_path / "state.json").exists():
        print(f"[resume] Loading from {resume_path}")
        all_params, opt_state, saved = _load_checkpoint(
            resume_path, all_params_template, opt_state_template)
        start_epoch  = saved["epoch"] + 1
        best_val_acc = saved["best_val_acc"]
        patience_ctr = saved["patience_ctr"]
        results      = saved.get("results", results)
        print(f"[resume] Epoch {start_epoch}  best={best_val_acc*100:.2f}%  patience={patience_ctr}")
    else:
        print("[train] No checkpoint — starting fresh.")

    @jax.jit
    def train_step(all_params, opt_state, frames, tasks, labels, rng):
        def loss_fn(params):
            obs      = {"image_primary": frames[:, -1:]}
            pad_mask = jnp.ones((frames.shape[0], 1), dtype=bool)
            out = octo_model.module.apply(
                {"params": params["backbone"]},
                obs, tasks, pad_mask,
                train=True,
                rngs={"dropout": rng},
                method="octo_transformer",
            )
            feat   = out["readout_action"].tokens[:, 0]
            logits = head.apply({"params": params["head"]}, feat, training=True)
            oh     = jax.nn.one_hot(labels, args.num_actions)
            loss   = -jnp.sum(oh * jax.nn.log_softmax(logits), axis=-1).mean()
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(all_params)
        updates, new_opt = tx.update(grads, opt_state, all_params)
        new_params = optax.apply_updates(all_params, updates)
        preds = jnp.argmax(logits, axis=-1)
        acc   = (preds == labels).mean()
        return new_params, new_opt, loss, acc

    def validate(all_params, val_ep, val_refs):
        preds_all, labels_all = [], []
        for s in range(0, len(val_refs), args.batch_size):
            batch_refs = val_refs[s: s + args.batch_size]
            frames, tasks, acts = make_batch_from_refs(val_ep, batch_refs, task_cache)
            obs      = {"image_primary": jnp.array(frames[:, -1:])}
            pad_mask = jnp.ones((frames.shape[0], 1), dtype=bool)
            out = octo_model.module.apply(
                {"params": all_params["backbone"]},
                obs, tasks, pad_mask,
                train=False,
                method="octo_transformer",
            )
            feat   = out["readout_action"].tokens[:, 0]
            logits = head.apply({"params": all_params["head"]}, feat, training=False)
            preds_all.extend(np.array(jnp.argmax(logits, -1)).tolist())
            labels_all.extend(acts.tolist())
        return np.mean(np.array(preds_all) == np.array(labels_all))

    print(f"\n[train] Epochs {start_epoch}→{args.epochs}  "
          f"lr_backbone={args.lr_backbone}  lr_head={args.lr_head}  "
          f"batch={args.batch_size}  num_actions={args.num_actions}")
    print("[train] First epoch includes JIT compilation — may take several minutes.")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        random.shuffle(trn_refs)
        trn_losses, trn_accs = [], []

        for s in range(0, len(trn_refs), args.batch_size):
            batch_refs = trn_refs[s: s + args.batch_size]
            if len(batch_refs) < 2:
                continue
            frames, tasks, acts = make_batch_from_refs(trn_ep, batch_refs, task_cache)
            key, rng = jax.random.split(key)
            all_params, opt_state, loss, acc = train_step(
                all_params, opt_state,
                jnp.array(frames), tasks, jnp.array(acts), rng,
            )
            trn_losses.append(float(loss))
            trn_accs.append(float(acc))

        val_acc = validate(all_params, val_ep, val_refs)
        trn_acc = float(np.mean(trn_accs))
        elapsed = time.time() - t0

        results["train_acc"].append(trn_acc)
        results["val_acc"].append(float(val_acc))

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"trn_acc={trn_acc:.4f}  val_acc={val_acc:.4f}  "
              f"loss={np.mean(trn_losses):.4f}  [{elapsed:.1f}s]")

        _save_checkpoint(ckpt_dir, all_params, opt_state, {
            "epoch":        epoch,
            "best_val_acc": float(best_val_acc),
            "patience_ctr": patience_ctr,
            "results":      results,
        })

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            patience_ctr = 0
            (out / "best_head.msgpack").write_bytes(
                serialization.to_bytes(all_params["head"]))
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"[early stop] No improvement for {args.patience} epochs.")
                break

    results["best_val_acc"] = float(best_val_acc)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] Best val acc: {best_val_acc*100:.2f}%  (random baseline: {100/args.num_actions:.1f}%)")
    print(f"[done] Results → {out}/results.json")


if __name__ == "__main__":
    main()
