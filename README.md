# TERM — Trustworthy Embodied Robot with Memory

**TERM** is a closed-loop robot learning framework built around **VERA** (Vision-Experience-Reasoning-Action), a five-stream Vision-Language-Action (VLA) policy that grounds action decisions in both language instructions and the robot's own experiential memory.

> Submitted to AAAI 2027 (anonymous review)

---

## Overview

Standard VLA policies are open-loop: they map observations to actions without any feedback about what actually happened. VERA closes this loop with two novel experiential streams:

- **E_act** — a natural-language narration of the action the robot just took (e.g., *"I pushed the object to the right"*), gated by a reward-conditioned MLP.
- **E_exp** — a verbalization of the outcome (e.g., *"The object moved 3 cm closer to the target; reward +0.12"*), giving the policy grounded spatial feedback at every step.

These streams re-enter the policy at the next step alongside frozen visual and language goal encodings, enabling the model to reason across time without external memory.

---

## Architecture

```
Stream 1 — Vision:         T=3 RGB frames → CLIP ViT-B/32 → V tokens
Stream 2 — Instruction:    language goal  → CLIP text enc → L_instr
Stream 3a — Action (E_act): prev action → vocab → reward gate → CLIP → E_act  [novel]
Stream 3b — Experience (E_exp): verbalize(r, Δd) → CLIP → E_exp               [novel]
Stream 4 — Action History: {action, action_vec, reward} × H → TemporalHistoryTransformer → H tokens

Token sequence: [ L_instr | E_act | E_exp | V_1…V_T | H_1…H_H | CLS ]

Fusion backbone (LLaMA-ViLT hybrid):
  • Bidirectional attention  • RMSNorm  • RoPE  • SwiGLU FFN
  • 6 layers, 8 heads, d=256

Action heads (from CLS):
  • Discrete:    3-layer MLP → argmax action bin
  • Continuous:  2-layer MLP + Tanh → action vector ∈ [-1,1]^d
```

**Total trainable parameters**: ~96M (backbone) + 641K (action heads)

---

## Repository Structure

```
TERM/Latest/
├── configs/               # YAML config files (Language-Table, CALVIN, ablations)
│   └── config.yaml        # default config with full documentation
├── data/                  # Dataset loading utilities
│   ├── trajectory_dataset.py
│   └── calvin_utils.py
├── docs/                  # Paper (AAAI 2027 submission), architecture diagrams, figures
├── envs/                  # Simulation environment wrappers
│   └── sim_env.py
├── evaluation/            # Rollout evaluation scripts
│   ├── evaluate_vera.py
│   ├── vera_calvin_policy.py
│   └── vera_language_table_policy.py
├── models/                # Core model definitions
│   ├── vera_model.py      # VERA five-stream VLA policy
│   └── vla_model.py       # Baseline VLA (no feedback streams)
├── scripts/               # Training launch scripts and utilities
├── training/              # SFT and RL trainers
│   ├── sft_trainer_vera.py
│   └── rl_trainer_vera.py
├── vendor/                # Vendored dependencies
│   └── language-table/    # Language-Table environment (Google DeepMind)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone <repo-url>
cd TERM/Latest

pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git

# Vendor the Language-Table environment
pip install -e vendor/language-table/
```

For CALVIN benchmark support, follow the [CALVIN installation guide](https://github.com/mees/calvin).

---

## Quick Start

### 1. Configure

Edit `configs/config.yaml` and set `data.episodes_path` to your dataset root:

```yaml
data:
  episodes_path: /path/to/language_table_data
  dataset_type:  language_table   # or "calvin"
```

### 2. Supervised Fine-Tuning (Behavioural Cloning)

```bash
python -m training.sft_trainer_vera --config configs/config.yaml
```

### 3. Online RL Fine-Tuning

```bash
python -m training.rl_trainer_vera --config configs/config.yaml \
    --checkpoint checkpoints/best_sft_vera.pt
```

### 4. Evaluation

```bash
# Language-Table
python scripts/run_language_table_rollout_eval.py \
    --checkpoint checkpoints/best_sft_vera.pt \
    --config configs/config.yaml

# CALVIN
python scripts/run_calvin_rollout_eval.py \
    --checkpoint checkpoints/best_sft_vera.pt \
    --config configs/calvin_vera_max.yaml
```

---

## Ablation Flags

VERA's streams can be individually disabled via config to reproduce any ablation from the paper:

| Ablation | Flag | Description |
|----------|------|-------------|
| A | `vera.use_lang_feedback: false` | Base VLA — no language feedback |
| B | `vera.use_temporal_history: false` | Flat positional history (no TemporalHistoryTransformer) |
| C | `vera.use_reward_gate: false` | No reward gate on E_act |
| D | `vera.alignment_loss_coef: 0.0` | No contrastive alignment loss |
| F | `vera.use_consequence_token: false` | Action narration only (no E_exp) |
| G | `vera.use_consequence_token: true` + no E_act | Outcome token only |

---

## Benchmarks

| Benchmark | Metric | VERA (full) | Base VLA |
|-----------|--------|-------------|----------|
| Language-Table | Success rate | See paper | — |
| CALVIN (D→D) | Avg. tasks completed / 5 | See paper | — |

Full numbers and ablation table in `docs/aaai2027_vera_submission.tex`.

---

## Citation

```bibtex
@inproceedings{vera2027,
  title     = {TERM: Trustworthy Embodied Robot with Memory via Closed-Loop Experiential Feedback},
  author    = {Anonymous},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2027},
  note      = {Under review}
}
```

---

## License

For research use only. License to be determined upon publication.
# TERM
