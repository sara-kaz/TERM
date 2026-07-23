#!/usr/bin/env python3
"""Verify alignment loss backprops into fusion / projection layers (not just log_temp)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.evaluate_vera import build_vera_from_cfg


def main():
    cfg = yaml.safe_load(open(REPO / "configs/calvin_config.yaml"))
    m = build_vera_from_cfg(cfg, "cpu")
    m.train()
    B, H, T = 8, 4, 3
    frames = torch.randn(B, T, 3, 224, 224)
    lang = torch.randint(0, 100, (B, 77))
    ah = torch.randint(0, 14, (B, H))
    rh = torch.rand(B, H)
    out = m(frames, lang, ah, rh)
    align = m.compute_alignment_loss(
        out["instr_token"],
        out.get("action_lang_token"),
        rh[:, -1],
        out.get("consequence_token"),
    )
    align.backward()
    grads = {n: (p.grad.abs().sum().item() if p.grad is not None else 0.0)
             for n, p in m.named_parameters() if p.requires_grad}
    fusion = sum(v for k, v in grads.items() if "fusion_transformer" in k)
    lang_proj = grads.get("lang_proj.0.weight", 0.0)
    act_proj = grads.get("action_lang_encoder.proj.0.weight", 0.0)
    cons_proj = grads.get("consequence_encoder.proj.0.weight", 0.0)
    log_temp = grads.get("alignment_module.log_temp", 0.0)
    print(f"align loss={align.item():.4f}")
    print(f"  fusion_transformer grad sum: {fusion:.4f}")
    print(f"  lang_proj grad:            {lang_proj:.4f}")
    print(f"  action_lang proj grad:     {act_proj:.4f}")
    print(f"  consequence proj grad:     {cons_proj:.4f}")
    print(f"  alignment log_temp grad:   {log_temp:.4f}")
    ok = lang_proj > 0 and act_proj > 0 and cons_proj > 0
    print("PASS" if ok else "FAIL — alignment still disconnected from trainable projections")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
