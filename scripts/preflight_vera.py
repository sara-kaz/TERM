#!/usr/bin/env python3
"""
Pre-flight gate before expensive VERA retrain / ablation runs.

Runs automated checks on everything that affects ablation validity and
embodied task success for CALVIN and Language-Table.

Usage:
  python scripts/preflight_vera.py
  python scripts/preflight_vera.py --calvin_path ~/calvin_task_D/task_D_D

Exit 0 = safe to launch full training. Exit 1 = fix failures first.
"""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FAILURES: list[str] = []
WARNINGS: list[str] = []


def ok(msg: str):
    print(f"  [OK]   {msg}")


def fail(msg: str):
    print(f"  [FAIL] {msg}")
    FAILURES.append(msg)


def warn(msg: str):
    print(f"  [WARN] {msg}")
    WARNINGS.append(msg)


def check_subprocess(script: str, args: list[str] | None = None) -> bool:
    cmd = [sys.executable, str(REPO / "scripts" / script)] + (args or [])
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode == 0:
        for line in out.strip().splitlines()[-3:]:
            print(f"         {line}")
        return True
    fail(f"{script} exited {r.returncode}")
    for line in out.strip().splitlines()[-8:]:
        print(f"         {line}")
    return False


def check_configs():
    print("\n── Config sanity ──")
    lt = yaml.safe_load(open(REPO / "configs/config.yaml"))
    cal = yaml.safe_load(open(REPO / "configs/calvin_config_ltdev.yaml"))

    lt_k = int(lt["model"].get("chunk_size", 1))
    cal_k = int(cal["model"].get("chunk_size", 1))
    if lt_k == 1:
        ok(f"LT chunk_size={lt_k}")
    else:
        fail(f"LT chunk_size={lt_k} (must be 1 until chunk rollout is implemented)")

    if cal_k == 1:
        ok(f"CALVIN chunk_size={cal_k}")
    else:
        warn(f"CALVIN chunk_size={cal_k} — prefer 1 for embodied eval")

    lt_crop = float(lt.get("data", {}).get("crop_factor", 1.0))
    if abs(lt_crop - 0.95) < 1e-6:
        ok(f"LT crop_factor={lt_crop}")
    else:
        warn(f"LT crop_factor={lt_crop} (official eval uses 0.95)")

    for name, cfg in [("LT", lt), ("CALVIN", cal)]:
        cl = float(cfg.get("training", {}).get("closed_loop_dropout", 0.0))
        if cl >= 0.25:
            ok(f"{name} closed_loop_dropout={cl}")
        else:
            warn(f"{name} closed_loop_dropout={cl} (recommend ≥0.35 for rollout robustness)")

    if cal.get("vera", {}).get("use_action_lang_feedback") is not False:
        ok("CALVIN has use_action_lang_feedback flag")
    else:
        warn("CALVIN ltdev config has use_action_lang_feedback=false")


def check_ablation_architectures():
    print("\n── Ablation architecture separation ──")
    from evaluation.evaluate_vera import build_vera_from_cfg

    base = yaml.safe_load(open(REPO / "configs/calvin_config_ltdev.yaml"))
    variants = {
        "full_vera": {},
        "bc_baseline": {
            "use_action_lang_feedback": False,
            "use_consequence_token": False,
            "use_temporal_history": False,
        },
        "no_act": {"use_action_lang_feedback": False},
        "no_exp": {"use_consequence_token": False},
        "no_lang": {
            "use_action_lang_feedback": False,
            "use_consequence_token": False,
        },
        "no_history_tf": {"use_temporal_history": False},
    }
    sigs = {}
    for name, vera_ov in variants.items():
        c = copy.deepcopy(base)
        c["vera"].update(vera_ov)
        m = build_vera_from_cfg(c, "cpu")
        m.cpu()
        sig = (
            m.use_action_lang_feedback,
            m.use_consequence_token,
            m.history_encoder.use_temporal,
            m.action_lang_encoder is not None,
            m.consequence_encoder is not None,
        )
        sigs[name] = sig
        n = sum(p.numel() for p in m.parameters())
        print(f"         {name:16s} params={n/1e6:.2f}M  sig={sig}")

    if sigs["no_act"] != sigs["no_lang"] and sigs["no_act"] != sigs["bc_baseline"]:
        ok("no_act is distinct from no_lang and bc_baseline")
    else:
        fail("no_act ablation collides with no_lang or bc_baseline")

    if sigs["no_exp"] != sigs["full_vera"]:
        ok("no_exp differs from full_vera")
    else:
        fail("no_exp identical to full_vera")

    if sigs["bc_baseline"] != sigs["full_vera"]:
        ok("bc_baseline differs from full_vera")
    else:
        fail("bc_baseline identical to full_vera")


def check_calvin_labels(calvin_path: str | None):
    print("\n── CALVIN label distribution ──")
    if not calvin_path or not Path(calvin_path).exists():
        warn("Skipping CALVIN label check (no --calvin_path)")
        return
    from data.trajectory_dataset import load_calvin

    eps = load_calvin(calvin_path, split="training")
    if not eps:
        fail("CALVIN load returned 0 episodes")
        return
    counts = [0] * 14
    n = 0
    for ep in eps[:200]:
        for a in ep["actions"]:
            if 0 <= int(a) < 14:
                counts[int(a)] += 1
                n += 1
    arm = sum(counts[:12])
    grip = counts[12] + counts[13]
    if n == 0:
        fail("No CALVIN actions sampled")
        return
    arm_frac = arm / n
    print(f"         arm classes 0-11: {arm} ({100*arm_frac:.1f}%)  gripper 12-13: {grip}")
    if arm_frac < 0.05:
        fail(f"CALVIN label collapse: only {100*arm_frac:.1f}% arm classes (expect >>5%)")
    elif arm_frac > 0.15:
        ok(f"CALVIN arm classes present ({100*arm_frac:.1f}%)")
    else:
        warn(f"CALVIN arm fraction low ({100*arm_frac:.1f}%) — verify discretization")


def check_alignment_gradients():
    print("\n── Alignment gradient flow ──")
    check_subprocess("verify_alignment_gradients.py")


def check_frame_alignment(calvin_path: str | None):
    print("\n── Train/rollout frame alignment ──")
    args = ["--config", str(REPO / "configs/calvin_config_ltdev.yaml")]
    if calvin_path and Path(calvin_path).exists():
        args += ["--calvin_path", calvin_path]
    check_subprocess("verify_rollout_train_alignment.py", args)


def check_ce_supervision():
    print("\n── CE supervision (primary head always trained) ──")
    from training.sft_trainer_vera import build_model

    cfg = yaml.safe_load(open(REPO / "configs/config.yaml"))
    cfg["model"]["chunk_size"] = 4  # stress test
    m = build_model(cfg)
    m.cpu()
    m.train()
    B, H, T, A = 4, 4, 3, 8
    frames = torch.randn(B, T, 3, 224, 224)
    lang = torch.randint(0, 100, (B, 77))
    ah = torch.randint(0, A, (B, H))
    rh = torch.rand(B, H)
    out = m(frames, lang, ah, rh)
    if "logits_chunk" not in out:
        fail("chunk_size=4 model missing logits_chunk head")
        return
    logits = out["logits"]
    logits_chunk = out["logits_chunk"]
    target = torch.randint(0, A, (B,))
    target_chunk = torch.randint(0, A, (B, 4))
    ce_p = torch.nn.functional.cross_entropy(logits, target)
    ce_c = torch.nn.functional.cross_entropy(
        logits_chunk.view(B * 4, A), target_chunk.view(B * 4)
    )
    loss = 0.5 * ce_p + 0.5 * ce_c
    loss.backward()
    ah_grad = m.action_bin_head[-1].weight.grad
    ch_grad = m.action_chunk_head.weight.grad
    if ah_grad is not None and ah_grad.abs().sum() > 0:
        ok("primary action_head receives gradient when chunk_size>1")
    else:
        fail("primary action_head has zero gradient with chunk_size>1")
    if ch_grad is not None and ch_grad.abs().sum() > 0:
        ok("action_chunk_head receives gradient")
    else:
        fail("action_chunk_head has zero gradient")


def check_bidirectional_fusion():
    print("\n── Bidirectional fusion (paper-critical) ──")
    import inspect
    from models import vera_model as vm
    src = inspect.getsource(vm.VERAModel.forward)
    if "fusion_transformer(sequence, attn_mask=None)" in src:
        ok("main fusion uses bidirectional attention (no causal mask)")
    elif "causal_mask" in src and "fusion_transformer(sequence, attn_mask=causal_mask)" in src:
        fail("main fusion still uses causal mask — E_act/E_exp blind to vision")
    else:
        warn("could not verify fusion attention mode from source")
    print("\n── Corrupt-consequence patch isolation ──")
    import models.vera_model as vm
    from scripts.run_calvin_ablations import _corrupt_consequence_ctx

    orig = vm.verbalize_consequence
    with _corrupt_consequence_ctx():
        corrupted = vm.verbalize_consequence(0.5, 0.0)
    if vm.verbalize_consequence is orig:
        ok("verbalize_consequence restored after context exit")
    else:
        fail("verbalize_consequence NOT restored — ablation matrix corrupted")
    if corrupted != orig(0.5, 0.0):
        ok("corrupt patch active inside context")
    else:
        warn("corrupt patch may not be working inside context")


def main():
    p = argparse.ArgumentParser(description="VERA pre-flight checks")
    p.add_argument("--calvin_path", default=None)
    p.add_argument(
        "--fast",
        action="store_true",
        help="Skip slow CALVIN full-dataset load (configs + model checks only)",
    )
    args = p.parse_args()

    calvin = None if args.fast else (args.calvin_path or str(Path.home() / "calvin_task_D/task_D_D"))

    print("=" * 60)
    print("  VERA PRE-FLIGHT (CALVIN + Language-Table)")
    print("=" * 60)

    check_configs()
    check_ablation_architectures()
    check_calvin_labels(calvin)
    check_alignment_gradients()
    check_frame_alignment(calvin)
    check_ce_supervision()
    check_bidirectional_fusion()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"  BLOCKED — {len(FAILURES)} failure(s):")
        for f in FAILURES:
            print(f"    • {f}")
        if WARNINGS:
            print(f"  ({len(WARNINGS)} warning(s))")
        sys.exit(1)
    print("  READY for full retrain")
    if WARNINGS:
        print(f"  ({len(WARNINGS)} warning(s) — review above)")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
