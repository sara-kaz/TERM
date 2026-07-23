"""
VLLA Ablation Study Runner
==========================
Trains or evaluates all ablation variants of VLLA and produces:
  1. JSON results file with mean ± std per metric per ablation
  2. Markdown table (printed to stdout, suitable for paper)
  3. CSV for plotting

This script supports two modes:
  --mode eval   : load a single trained checkpoint and evaluate all ablations
                  by disabling modules at inference time (fast; ~10 min)
  --mode train  : fully re-train each ablation from scratch (slow; ~hours)
                  Required for a fully rigorous ablation — NeurIPS standard

Usage
-----
# Fast eval-only ablations (uses a pretrained full-model checkpoint)
python scripts/run_ablations.py \\
    --config configs/config.yaml \\
    --checkpoint checkpoints/rl/best_rl_vera.pt \\
    --mode eval --episodes 100 --seeds 5

# Full train ablations (each variant trained independently from scratch)
python scripts/run_ablations.py \\
    --config configs/config.yaml \\
    --mode train --seeds 3
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ── Ablation definitions ──────────────────────────────────────────────────────
# Each entry: (display_name, vera_config_overrides)
# The "Full VLLA" row uses the config as-is (no overrides).

ABLATIONS = [
    # ── Core contribution ablations ──────────────────────────────────────────
    ("Full VLLA (ours)",
     {}),

    ("A — No language feedback (base VLA)",
     {"use_action_lang_feedback": False, "use_consequence_token": False}),

    ("B — No temporal history",
     {"use_temporal_history": False}),

    ("C — No reward gate on action token",
     {"use_reward_gate": False}),

    ("D — No contrastive alignment loss",
     {"alignment_loss_coef": 0.0}),

    # ── Consequence token ablations (new contribution) ────────────────────────
    ("F — Action token only (no consequence)",
     {"use_consequence_token": False}),

    ("G — Consequence token only (no action lang)",
     {"use_action_lang_feedback": False, "use_consequence_token": True}),

    # ── Minimal baseline ─────────────────────────────────────────────────────
    ("E — Minimal (no lang, no history)",
     {"use_action_lang_feedback": False,
      "use_consequence_token": False,
      "use_temporal_history": False}),
]


# ── Eval-mode ablation ────────────────────────────────────────────────────────

def run_eval_ablations(cfg: dict, checkpoint: str,
                        num_episodes: int, num_seeds: int) -> dict:
    """
    Load the same checkpoint once per ablation, disable components via
    config overrides, and evaluate. Fast but approximate — the shared
    weights were trained with all components enabled.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from evaluation.evaluate_vera import build_vera_from_cfg, load_checkpoint, evaluate_once

    device  = "cuda"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "mps" \
                 if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() \
                 else "cpu"
    except Exception:
        pass

    all_results = {}

    for name, overrides in ABLATIONS:
        print(f"\n{'='*60}")
        print(f"  Ablation: {name}")
        print(f"  Overrides: {overrides if overrides else '(none — full model)'}")
        print(f"{'='*60}")

        abl_cfg = copy.deepcopy(cfg)
        for k, v in overrides.items():
            abl_cfg["vera"][k] = v

        # training alignment_loss_coef has no effect at eval; skip model rebuild
        model = build_vera_from_cfg(abl_cfg, device)
        try:
            load_checkpoint(model, checkpoint, device)
        except RuntimeError:
            import torch
            ckpt  = torch.load(checkpoint, map_location=device)
            state = ckpt.get("model_state", ckpt)
            model.load_state_dict(state, strict=False)
            model.eval()

        seed_returns, seed_successes, seed_lengths, seed_entropies = [], [], [], []
        for s in range(num_seeds):
            t0 = time.time()
            res = evaluate_once(model, abl_cfg, num_episodes=num_episodes,
                                deterministic=True, seed=s * 17)
            elapsed = time.time() - t0
            seed_returns.append(res["mean_return"])
            seed_successes.append(res["success_rate"])
            seed_lengths.append(res["mean_length"])
            seed_entropies.append(res["mean_entropy"])
            print(f"  Seed {s+1}/{num_seeds} | return={res['mean_return']:.4f} "
                  f"success={res['success_rate']*100:.1f}% "
                  f"len={res['mean_length']:.1f} ({elapsed:.1f}s)")

        all_results[name] = {
            "overrides": overrides,
            "mean_return":  float(np.mean(seed_returns)),
            "std_return":   float(np.std(seed_returns)),
            "mean_success": float(np.mean(seed_successes)),
            "std_success":  float(np.std(seed_successes)),
            "mean_length":  float(np.mean(seed_lengths)),
            "std_length":   float(np.std(seed_lengths)),
            "mean_entropy": float(np.mean(seed_entropies)),
            "std_entropy":  float(np.std(seed_entropies)),
            "raw_returns":  seed_returns,
            "raw_successes": seed_successes,
        }

    return all_results


# ── Train-mode ablation ───────────────────────────────────────────────────────

def run_train_ablations(cfg: dict, num_seeds: int, out_dir: Path) -> dict:
    """
    Re-train each ablation variant from scratch with `num_seeds` seeds.
    This is the fully rigorous approach required for NeurIPS.
    Each variant gets its own output subdirectory.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.sft_trainer_vera import sft_train
    from training.rl_trainer_vera  import rl_train
    from evaluation.evaluate_vera  import build_vera_from_cfg, load_checkpoint, evaluate_once
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results = {}

    for name, overrides in ABLATIONS:
        print(f"\n{'='*60}")
        print(f"  Training ablation: {name}")
        print(f"{'='*60}")
        slug = name.split("—")[0].strip().replace(" ", "_").lower()

        seed_returns, seed_successes = [], []

        for s in range(num_seeds):
            abl_cfg = copy.deepcopy(cfg)
            for k, v in overrides.items():
                abl_cfg["vera"][k] = v
            seed_out = out_dir / slug / f"seed{s}"
            seed_out.mkdir(parents=True, exist_ok=True)
            abl_cfg["training"]["output_dir"] = str(seed_out)

            import torch as _torch
            _torch.manual_seed(s * 31)
            np.random.seed(s * 31)

            print(f"  Seed {s+1}/{num_seeds} — SFT …")
            sft_train(abl_cfg)

            print(f"  Seed {s+1}/{num_seeds} — RL …")
            rl_train(abl_cfg)

            ckpt = seed_out / "rl" / "best_rl_vera.pt"
            if not ckpt.exists():
                ckpt = seed_out / "best_sft_vera.pt"
            model = build_vera_from_cfg(abl_cfg, device)
            load_checkpoint(model, str(ckpt), device)
            res = evaluate_once(model, abl_cfg, num_episodes=50,
                                deterministic=True, seed=s)
            seed_returns.append(res["mean_return"])
            seed_successes.append(res["success_rate"])
            print(f"    return={res['mean_return']:.4f}  "
                  f"success={res['success_rate']*100:.1f}%")

        all_results[name] = {
            "overrides": overrides,
            "mean_return":  float(np.mean(seed_returns)),
            "std_return":   float(np.std(seed_returns)),
            "mean_success": float(np.mean(seed_successes)),
            "std_success":  float(np.std(seed_successes)),
            "raw_returns":  seed_returns,
            "raw_successes": seed_successes,
        }

    return all_results


# ── Table printer ─────────────────────────────────────────────────────────────

def print_markdown_table(results: dict):
    """Print a NeurIPS-style ablation table in GitHub-flavoured markdown."""
    # Find the full-model row for delta computation
    full_return  = results.get("Full VLLA (ours)", {}).get("mean_return", 0.0)
    full_success = results.get("Full VLLA (ours)", {}).get("mean_success", 0.0)

    hdr = (f"| {'Method':<45} | {'Return (mean±std)':>20} "
           f"| {'Success (mean±std)':>22} | {'ΔReturn':>9} | {'ΔSuccess':>10} |")
    sep = f"|{'-'*47}|{'-'*22}|{'-'*24}|{'-'*11}|{'-'*12}|"

    print("\n## Table 1 — VLLA Ablation Study\n")
    print(hdr)
    print(sep)

    for name, vals in results.items():
        mu_r   = vals["mean_return"]
        std_r  = vals["std_return"]
        mu_s   = vals["mean_success"]
        std_s  = vals["std_success"]
        delta_r = mu_r - full_return   if not name.startswith("Full") else 0.0
        delta_s = mu_s - full_success  if not name.startswith("Full") else 0.0

        marker = " †" if name.startswith("Full") else ""
        ret_str = f"{mu_r:.3f} ± {std_r:.3f}"
        suc_str = f"{mu_s*100:.1f} ± {std_s*100:.1f}%"
        dr_str  = f"{delta_r:+.3f}" if not name.startswith("Full") else "—"
        ds_str  = f"{delta_s*100:+.1f}%" if not name.startswith("Full") else "—"

        print(f"| {name+marker:<45} | {ret_str:>20} | {suc_str:>22} | {dr_str:>9} | {ds_str:>10} |")

    print("\n† proposed method")
    print("ΔReturn / ΔSuccess = difference from Full VLLA (negative = ablation hurts)")


def save_csv(results: dict, path: Path):
    lines = ["method,mean_return,std_return,mean_success,std_success"]
    for name, vals in results.items():
        lines.append(
            f"\"{name}\",{vals['mean_return']:.4f},{vals['std_return']:.4f},"
            f"{vals['mean_success']:.4f},{vals['std_success']:.4f}"
        )
    path.write_text("\n".join(lines))
    print(f"CSV saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VERA ablation study runner")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Pretrained VLLA checkpoint (required for --mode eval)")
    parser.add_argument("--mode",       choices=["eval", "train"], default="eval",
                        help="eval=fast inference-time ablation; train=full retrain per variant")
    parser.add_argument("--episodes",   type=int, default=100,
                        help="Episodes per seed per ablation (eval mode only)")
    parser.add_argument("--seeds",      type=int, default=5,
                        help="Number of random seeds to average")
    parser.add_argument("--out",        default="results/ablations",
                        help="Output directory for results")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "eval":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for --mode eval")
        results = run_eval_ablations(cfg, args.checkpoint,
                                     num_episodes=args.episodes,
                                     num_seeds=args.seeds)
    else:
        results = run_train_ablations(cfg, num_seeds=args.seeds, out_dir=out_dir)

    # Save JSON
    json_path = out_dir / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # Print table
    print_markdown_table(results)

    # Save CSV
    save_csv(results, out_dir / "ablation_results.csv")


if __name__ == "__main__":
    main()
