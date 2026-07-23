"""
CALVIN comparison runner for VERA — **same experiment matrix as Language-Table.**

Protocol (matches `scripts/run_language_table_comparisons.py` in sibling setups):
  - 12 numbered conditions (01–12), same vera overrides / patches
  - seeds: 42, 123, 456
  - epochs / batch / lr / val_frac / eval episodes enforced via `_common_cfg`
  - skip completed seeds unless `--force`
  - optional `--run-eval` (100 episodes, dummy SimEnv — same caveat as LT)

Usage
-----
  # 1) Download & extract CALVIN D→D (see scripts/download_calvin_task_d_d.sh)

  # 2) Run full matrix on 2 GPUs (default), LT-dev-style losses (calvin_config_ltdev.yaml)
  python scripts/run_calvin_comparisons.py \\
      --calvin_path /path/to/task_D_D \\
      --config configs/calvin_config_ltdev.yaml \\
      --out results/calvin_comparisons_ltdev \\
      --gpus 0,1

  # Resume condition 06+, optional eval rollouts after each trained seed:
  python scripts/run_calvin_comparisons.py --calvin_path ... --start-from 6 --run-eval

  # Smoke test (2 epochs, synthetic data — no CALVIN files required):
  python scripts/run_calvin_comparisons.py --calvin_path /tmp/absent \\
      --dry-run --start-from 1 --gpus ""
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

SEEDS = [42, 123, 456]


def _common_cfg(cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    out["training"]["epochs"] = 100
    out["training"]["batch_size"] = 32
    out["training"]["lr"] = 3e-4
    out["training"]["val_fraction"] = 0.2
    out["training"]["early_stopping_patience"] = out["training"].get(
        "early_stopping_patience", 15
    )
    out["eval"]["num_episodes"] = 100
    return out


@contextlib.contextmanager
def _patch_action_stream_off(enable: bool):
    if not enable:
        yield
        return
    import torch
    import models.vera_model as vm

    orig_forward = vm.ActionLanguageFeedbackEncoder.forward

    def _forward_zero(self, prev_action_idx, prev_reward):
        b = prev_action_idx.size(0)
        d_model = (
            self.proj[-1].weight.shape[0]
            if hasattr(self.proj[-1], "weight")
            else 256
        )
        token = torch.zeros(
            (b, 1, d_model), device=prev_action_idx.device, dtype=torch.float32
        )
        raw = torch.zeros(
            (b, 512), device=prev_action_idx.device, dtype=torch.float32
        )
        return token, raw

    vm.ActionLanguageFeedbackEncoder.forward = _forward_zero
    try:
        yield
    finally:
        vm.ActionLanguageFeedbackEncoder.forward = orig_forward


@contextlib.contextmanager
def _patch_consequence_text(mode: str):
    import models.vera_model as vm

    orig = vm.verbalize_consequence
    rng = np.random.default_rng(seed=0)
    random_phrases = [
        "The weather is sunny today",
        "A bicycle is parked outside",
        "There are three apples on the table",
        "The train arrives at platform 4",
        "Music plays in the background",
        "A book is open on the desk",
    ]

    if mode == "normal":
        yield
        return
    if mode == "corrupted":
        vm.verbalize_consequence = lambda *args, **kwargs: str(rng.choice(random_phrases))
    elif mode == "reward_only":
        vm.verbalize_consequence = lambda reward, delta_dist=None: orig(reward, None)
    else:
        raise ValueError(f"Unknown consequence mode: {mode}")
    try:
        yield
    finally:
        vm.verbalize_consequence = orig


CONDITIONS = [
    ("01_vera_full", "VERA (full)", {}, False, "normal"),
    (
        "02_bc_sft",
        "BC/SFT baseline",
        {
            "use_action_lang_feedback": False,
            "use_consequence_token": False,
            "use_temporal_history": False,
        },
        False,
        "normal",
    ),
    (
        "03_bc_history_only",
        "BC + history only (no language streams)",
        {"use_action_lang_feedback": False, "use_consequence_token": False},
        False,
        "normal",
    ),
    (
        "04_no_history_baseline",
        "No-history baseline (streams 3a+3b only)",
        {"use_temporal_history": False},
        False,
        "normal",
    ),
    ("05_no_e_exp", "No E_exp (Stream 3b off)", {"use_consequence_token": False}, False, "normal"),
    ("06_no_e_act", "No E_act (Stream 3a off)", {"use_action_lang_feedback": False}, False, "normal"),
    (
        "07_no_language_feedback",
        "No language feedback (3a+3b off)",
        {"use_action_lang_feedback": False, "use_consequence_token": False},
        False,
        "normal",
    ),
    ("08_no_history_tf", "No temporal history TF (Stream 4 sub-transformer off)", {"use_temporal_history": False}, False, "normal"),
    ("09_no_reward_gate", "No reward gate", {"use_reward_gate": False}, False, "normal"),
    (
        "10_no_dual_head",
        "No dual head / regression branch",
        {"regression_loss_coef": 0.0},
        False,
        "normal",
    ),
    ("11_corrupted_consequence", "Corrupted consequence text", {}, False, "corrupted"),
    (
        "12_reward_only_consequence",
        "Reward-only consequence verbalization",
        {},
        False,
        "reward_only",
    ),
]


def _seed_everything(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run(args):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from training.sft_trainer_vera import sft_train
        from evaluation.evaluate_vera import build_vera_from_cfg, evaluate_once, load_checkpoint
    except ModuleNotFoundError as e:
        if e.name == "clip":
            raise SystemExit(
                "Missing dependency: `clip`.\n"
                "Install with:\n"
                "  pip install git+https://github.com/openai/CLIP.git\n"
                "Then re-run."
            )
        raise

    with open(args.config) as f:
        base_cfg = _common_cfg(yaml.safe_load(f))

    if args.num_workers is not None:
        base_cfg["training"]["num_workers"] = int(args.num_workers)

    base_cfg["data"]["dataset_type"] = "calvin"
    base_cfg["data"]["episodes_path"] = args.calvin_path

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {}

    for idx, (slug, name, vera_overrides, action_off, consequence_mode) in enumerate(
        CONDITIONS, start=1
    ):
        if idx < args.start_from:
            continue

        print(f"\n{'=' * 70}\n[{idx:02d}] {name}\n{'=' * 70}")
        summary[slug] = {"name": name, "seeds": {}, "vera_overrides": vera_overrides}

        for seed in SEEDS:
            seed_dir = out_root / slug / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = seed_dir / "best_sft_vera.pt"
            log_path = seed_dir / "sft_vera_log.json"

            if ckpt_path.exists() and log_path.exists() and not args.force:
                print(f"  [skip] seed={seed} exists")
            else:
                cfg = copy.deepcopy(base_cfg)
                cfg["training"]["seed"] = seed
                cfg["training"]["output_dir"] = str(seed_dir)
                for k, v in vera_overrides.items():
                    cfg["vera"][k] = v

                if args.dry_run:
                    cfg["training"]["epochs"] = 2
                    cfg["data"]["synthetic_episodes"] = 20
                    cfg["data"]["allow_synthetic"] = True
                    print("  [dry_run] epochs=2, synthetic data")

                _seed_everything(seed)
                t0 = time.time()
                with (
                    _patch_action_stream_off(action_off),
                    _patch_consequence_text(consequence_mode),
                ):
                    sft_train(cfg)
                print(f"  seed={seed} train done in {(time.time() - t0) / 60:.1f} min")

            best_val = 0.0
            if log_path.exists():
                with open(log_path) as f:
                    rows = json.load(f)
                if rows:
                    best_val = float(max(r.get("val_acc", 0.0) for r in rows))

            run_stats = {"best_val_acc": best_val}

            if args.run_eval and ckpt_path.exists():
                cfg = copy.deepcopy(base_cfg)
                for k, v in vera_overrides.items():
                    cfg["vera"][k] = v
                cfg["training"]["seed"] = seed
                model = build_vera_from_cfg(cfg, _device())
                load_checkpoint(model, str(ckpt_path), _device())
                _seed_everything(seed)
                with (
                    _patch_action_stream_off(action_off),
                    _patch_consequence_text(consequence_mode),
                ):
                    eval_res = evaluate_once(
                        model,
                        cfg,
                        num_episodes=cfg["eval"]["num_episodes"],
                        deterministic=True,
                        seed=seed,
                    )
                run_stats["eval"] = eval_res

            summary[slug]["seeds"][str(seed)] = run_stats

        vals = [v["best_val_acc"] for v in summary[slug]["seeds"].values()]
        if vals:
            summary[slug]["mean_val_acc"] = float(np.mean(vals))
            summary[slug]["std_val_acc"] = float(np.std(vals))
            print(
                f"  => val_acc mean±std: {summary[slug]['mean_val_acc']:.4f}"
                f" ± {summary[slug]['std_val_acc']:.4f}"
            )

    out_file = out_root / "calvin_comparison_summary.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="CALVIN VERA comparison runner (matches Language-Table matrix)"
    )
    parser.add_argument(
        "--config",
        default="configs/calvin_config_ltdev.yaml",
        help="Base YAML (default: LT-dev-tuned CALVIN preset)",
    )
    parser.add_argument(
        "--calvin_path",
        required=True,
        help="task_D_D root (training/, validation/). Ignored paths → synthetic only in dry-run.",
    )
    parser.add_argument(
        "--out",
        default="results/calvin_comparisons_ltdev",
        help="Results + checkpoints subtree per condition/seed",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="After training, roll out eval.num_episodes (dummy env)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="1-based condition index [01–12] to resume from",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if best_sft_vera.pt exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="2 epochs, synthetic CALVIN-like episodes (sanity check)",
    )
    parser.add_argument(
        "--gpus",
        default="0,1",
        help="CUDA_VISIBLE_DEVICES (default: 0,1). Use empty string for CPU.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        metavar="N",
        help="Override training.num_workers (use 0 to avoid multiprocessing in restricted sandboxes)",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print(f"[gpu] CUDA_VISIBLE_DEVICES={args.gpus!r}")
    run(args)


if __name__ == "__main__":
    main()
