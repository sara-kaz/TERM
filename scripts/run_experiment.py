"""
VLLA Full Experiment Runner
============================
Single entrypoint that runs the complete NeurIPS experiment pipeline:

  Stage 1 — SFT  : behavioural cloning on demonstration data
  Stage 2 — RL   : online fine-tuning with feedback loop active
  Stage 3 — Eval : multi-seed evaluation of the trained policy
  Stage 4 — Ablations: eval-mode ablation comparison table

Each stage can be skipped independently via --skip-* flags.
Multi-seed runs are supported: each seed gets its own output subdirectory
and results are aggregated at the end.

Usage
-----
# Full pipeline, 3 seeds, MetaWorld reach task
python scripts/run_experiment.py \\
    --config configs/config.yaml \\
    --seeds 3 \\
    --env metaworld-reach-v2

# Skip SFT (e.g. you already have a BC checkpoint)
python scripts/run_experiment.py \\
    --config configs/config.yaml \\
    --seeds 3 \\
    --skip-sft \\
    --sft-checkpoint checkpoints/best_sft_vera.pt

# BabyAI (cheap, no GPU needed, good for debugging feedback loop)
python scripts/run_experiment.py \\
    --config configs/config.yaml \\
    --seeds 5 \\
    --env BabyAI-GoToLocal-v0 \\
    --rl-epochs 50 --eval-episodes 200
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml


# ── helpers ───────────────────────────────────────────────────────────────────

def _device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _set_seed(s: int):
    import torch, random
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)


def _merge(cfg: dict, overrides: dict) -> dict:
    """Shallow-merge override dict into cfg (modifies copy)."""
    out = copy.deepcopy(cfg)
    for k, v in overrides.items():
        keys = k.split(".")
        d = out
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = v
    return out


# ── stage runners ─────────────────────────────────────────────────────────────

def stage_sft(cfg: dict, seed_dir: Path, seed: int) -> Path:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.sft_trainer_vera import sft_train
    import torch

    _set_seed(seed)
    cfg["training"]["output_dir"] = str(seed_dir)
    print(f"\n[SFT] seed={seed}  output={seed_dir}")
    t0 = time.time()
    sft_train(cfg)
    ckpt = seed_dir / "best_sft_vera.pt"
    print(f"[SFT] done in {time.time()-t0:.0f}s  checkpoint={ckpt}")
    return ckpt


def stage_rl(cfg: dict, seed_dir: Path, seed: int,
             sft_ckpt: Optional[Path] = None) -> Path:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.rl_trainer_vera import rl_train
    import torch, shutil

    _set_seed(seed)

    # Point the RL trainer at the SFT checkpoint
    if sft_ckpt is not None and sft_ckpt.exists():
        target = seed_dir / "best_sft_vera.pt"
        if not target.exists():
            shutil.copy(sft_ckpt, target)

    cfg["training"]["output_dir"] = str(seed_dir)
    print(f"\n[RL] seed={seed}  output={seed_dir}")
    t0 = time.time()
    rl_train(cfg)
    ckpt = seed_dir / "rl" / "best_rl_vera.pt"
    print(f"[RL] done in {time.time()-t0:.0f}s  checkpoint={ckpt}")
    return ckpt


def stage_eval(cfg: dict, ckpt: Path, num_episodes: int, seed: int) -> dict:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from evaluation.evaluate_vera import build_vera_from_cfg, load_checkpoint, evaluate_once
    import torch

    device = _device()
    model  = build_vera_from_cfg(cfg, device)
    try:
        load_checkpoint(model, str(ckpt), device)
    except RuntimeError:
        ckpt2  = torch.load(str(ckpt), map_location=device)
        state  = ckpt2.get("model_state", ckpt2)
        model.load_state_dict(state, strict=False)
        model.eval()

    return evaluate_once(model, cfg, num_episodes=num_episodes,
                         deterministic=True, seed=seed * 7)


def stage_ablations(cfg: dict, ckpt: Path, num_episodes: int, num_seeds: int,
                    out_dir: Path):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import importlib.util, subprocess
    abl_script = Path(__file__).parent / "run_ablations.py"
    cmd = [
        sys.executable, str(abl_script),
        "--config",     str(Path(__file__).parent.parent / "configs" / "config.yaml"),
        "--checkpoint", str(ckpt),
        "--mode",       "eval",
        "--episodes",   str(num_episodes),
        "--seeds",      str(num_seeds),
        "--out",        str(out_dir / "ablations"),
    ]
    print(f"\n[Ablations] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate_seeds(per_seed: list) -> dict:
    """Aggregate a list of evaluate_once result dicts → mean ± std summary."""
    keys = ["mean_return", "success_rate", "mean_length", "mean_entropy"]
    out  = {}
    for k in keys:
        vals = [r[k] for r in per_seed]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"]  = float(np.std(vals))
    return out


# ── main ─────────────────────────────────────────────────────────────────────

# Needed for Optional annotation at top of file before torch import
from typing import Optional


def main():
    parser = argparse.ArgumentParser(description="VERA full experiment runner")
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--seeds",     type=int, default=3)
    parser.add_argument("--out",       default="results/experiment",
                        help="Root output directory")

    # Environment override
    parser.add_argument("--env",       default=None,
                        help="Override env_id (e.g. metaworld-reach-v2, "
                             "BabyAI-GoToLocal-v0, dummy)")

    # Stage skips
    parser.add_argument("--skip-sft",       action="store_true")
    parser.add_argument("--skip-rl",        action="store_true")
    parser.add_argument("--skip-eval",      action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")

    # Override a pre-trained SFT checkpoint (used when --skip-sft)
    parser.add_argument("--sft-checkpoint", default=None)

    # Hyperparameter overrides
    parser.add_argument("--sft-epochs",  type=int, default=None)
    parser.add_argument("--rl-epochs",   type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--abl-episodes",  type=int, default=50)

    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.env:
        base_cfg["env"]["env_id"] = args.env
    if args.sft_epochs:
        base_cfg["training"]["epochs"] = args.sft_epochs
    if args.rl_epochs:
        base_cfg["rl"]["epochs"] = args.rl_epochs

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  VLLA Experiment Runner")
    print(f"  Seeds: {args.seeds}  |  Env: {base_cfg['env']['env_id']}")
    print(f"  Output: {out_dir.resolve()}")
    print("=" * 70)

    seed_eval_results = []
    best_ckpt = None   # checkpoint from last seed (used for ablations)

    for seed in range(args.seeds):
        seed_dir = out_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        cfg = copy.deepcopy(base_cfg)

        print(f"\n{'─'*70}")
        print(f"  SEED {seed+1}/{args.seeds}")
        print(f"{'─'*70}")

        # ── Stage 1: SFT ──────────────────────────────────────────────────
        if args.skip_sft:
            if args.sft_checkpoint:
                sft_ckpt = Path(args.sft_checkpoint)
            else:
                sft_ckpt = seed_dir / "best_sft_vera.pt"
            print(f"[SFT] skipped — using {sft_ckpt}")
        else:
            sft_ckpt = stage_sft(cfg, seed_dir, seed)

        # ── Stage 2: RL ────────────────────────────────────────────────────
        if args.skip_rl:
            rl_ckpt = seed_dir / "rl" / "best_rl_vera.pt"
            print(f"[RL] skipped — using {rl_ckpt}")
        else:
            rl_ckpt = stage_rl(cfg, seed_dir, seed, sft_ckpt)

        best_ckpt = rl_ckpt if rl_ckpt.exists() else sft_ckpt

        # ── Stage 3: Eval ──────────────────────────────────────────────────
        if not args.skip_eval and best_ckpt.exists():
            print(f"\n[Eval] seed={seed} episodes={args.eval_episodes} …")
            res = stage_eval(cfg, best_ckpt, args.eval_episodes, seed)
            seed_eval_results.append(res)
            print(f"  return={res['mean_return']:.4f}  "
                  f"success={res['success_rate']*100:.1f}%  "
                  f"len={res['mean_length']:.1f}")
            with open(seed_dir / "eval_result.json", "w") as f:
                json.dump(res, f, indent=2)

    # ── Aggregate across seeds ─────────────────────────────────────────────
    if seed_eval_results:
        summary = aggregate_seeds(seed_eval_results)
        summary["num_seeds"]    = args.seeds
        summary["num_episodes"] = args.eval_episodes
        summary["env_id"]       = base_cfg["env"]["env_id"]

        print(f"\n{'='*70}")
        print("  FINAL RESULTS  (mean ± std across seeds)")
        print(f"{'='*70}")
        for k in ["mean_return", "success_rate", "mean_length", "mean_entropy"]:
            mu  = summary[f"{k}_mean"]
            std = summary[f"{k}_std"]
            print(f"  {k:<22}: {mu:.4f} ± {std:.4f}")

        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {out_dir / 'summary.json'}")

    # ── Stage 4: Ablations ─────────────────────────────────────────────────
    if not args.skip_ablations and best_ckpt is not None and best_ckpt.exists():
        stage_ablations(base_cfg, best_ckpt,
                        num_episodes=args.abl_episodes,
                        num_seeds=min(3, args.seeds),
                        out_dir=out_dir)

    print("\n[run_experiment] All stages complete.")


if __name__ == "__main__":
    main()
