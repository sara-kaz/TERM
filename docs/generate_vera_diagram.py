"""
VERA Architecture Diagram — Accurate Final Version
===================================================
Every component reflects the actual vera_model.py code exactly.

Five streams (top → bottom):
  1. Vision          — T=3 frames → CLIP Image → vis_proj(512→256) + RMSNorm
  2. Instruction     — CLIP Text  → lang_proj(512→256) + RMSNorm
  3a Action Narration— vocab[a_{t-1}] → CLIP Text → reward gate σ(MLP(r)) → proj → E_act
  3b Experience      — verbalize_consequence(r,Δd) → CLIP Text → proj → E_exp  [NEW]
  4. Action History  — 3 signals: Embed(idx) ⊕ σ(MLP(r))·Linear(vec) ⊕ Linear(r)
                        → fusion Linear(3d→d) + SinPE + TemporalHistoryTF (×2 LLaMA)

Token sequence:  [ L_instr | E_act | E_exp | V₁ V₂ V₃ | H₁ H₂ H₃ H₄ | CLS ]
(+ViLT type embeddings: type 0=instr/CLS, 1=act, 2=vis, 3=hist, 4=exp)

LLaMA Fusion: 6 × (RMSNorm → MHA+RoPE → ⊕ → RMSNorm → SwiGLU → ⊕ → Dropout)
Causal mask — CLS at position -1 attends to all 10 preceding tokens.

Dual head (from CLS):
  Branch 1 Discrete:   RMSNorm → Linear(d→d) → SiLU → Drop → Linear(d→d/2) → SiLU → Drop → Linear → argmax → aₜ
  Branch 2 Regression: RMSNorm → Linear(d→d/2) → SiLU → Drop → Linear(d/2→action_dim) → Tanh → v̂ₜ  [NEW]

Saves: docs/VERA.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Colour palette ─────────────────────────────────────────────────────────────
C_VIS  = "#1565C0"   # Vision          — deep blue
C_INS  = "#1B5E20"   # Instruction     — deep green
C_ACT  = "#BF360C"   # Action Narration— burnt orange
C_EXP  = "#283593"   # Experience      — indigo
C_HIS  = "#4A148C"   # Action History  — deep purple
C_CLIP = "#37474F"   # Frozen CLIP     — dark slate
C_GATE = "#E65100"   # Reward gate     — deep orange
C_VERB = "#004D40"   # verbalize box   — dark teal
C_FUSE = "#006064"   # LLaMA Fusion    — dark cyan
C_DISC = "#4E342E"   # Discrete head   — dark brown
C_REG  = "#283593"   # Regression head — indigo
C_LOOP = "#F57F17"   # Feedback loop   — amber
C_NEW  = "#B71C1C"   # [NEW] badge     — red
C_PASS = "#78909C"   # Pass-through    — blue-grey
C_DARK = "#212121"
C_BG   = "#FAFAFA"
C_GRID = "#CFD8DC"

# ── Canvas ─────────────────────────────────────────────────────────────────────
FW, FH = 36, 14.5
fig, ax = plt.subplots(figsize=(FW, FH), dpi=150)
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

# Column x-boundaries: [RowLabel | Encode | Gate/Verbalize | Project | Tokens | LLaMA | Head]
CX  = [2.80, 5.80, 8.40, 11.00, 13.50, 18.80, 35.60]
MID = [(CX[i] + CX[i+1]) / 2 for i in range(6)]

# Row y-centres (top → bottom) + CLS row below
ROW_Y  = [12.60, 11.10, 9.60, 8.10, 6.50]
CLS_Y  = 5.00
ROW_H  = 0.82
TOP_Y  = 13.45   # column header y

STREAM_LABELS = [
    "1 — Vision",
    "2 — Instruction",
    "3a — Action\n    Narration",
    "3b — Experience\n      [E in VERA]",
    "4 — Action\n    History",
]
STREAM_COLORS = [C_VIS, C_INS, C_ACT, C_EXP, C_HIS]

# ── Helpers ────────────────────────────────────────────────────────────────────
def box(cx, cy, w, h, fc, txt, fs=7.0, tc="white", bold=False,
        ec=None, lw=1.3, ls="-", alpha=1.0, zo=3):
    ec = ec or C_DARK
    pad = min(0.08, h * 0.08, w * 0.03)
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad={pad:.3f},rounding_size=0.10",
        fc=fc, ec=ec, lw=lw, ls=ls, alpha=alpha, zorder=zo,
    ))
    ax.text(cx, cy, txt, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal",
            zorder=zo + 1, multialignment="center", linespacing=1.25)

def arr(x0, y0, x1, y1=None, col=C_DARK, lw=1.5):
    y1 = y0 if y1 is None else y1
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=lw, mutation_scale=11), zorder=7)

def badge(cx, cy):
    bw, bh = 0.44, 0.23
    ax.add_patch(FancyBboxPatch((cx - bw/2, cy - bh/2), bw, bh,
                                boxstyle="round,pad=0.03",
                                fc=C_NEW, ec="none", zorder=12))
    ax.text(cx, cy, "NEW", ha="center", va="center",
            fontsize=5.2, color="white", fontweight="bold", zorder=13)

def divider(x):
    ax.plot([x, x], [0.35, TOP_Y + 0.08], color=C_GRID, lw=0.9, ls="--", zorder=1)

# ════════════════════════════════════════════════════════════════════════════════
#  TITLE
# ════════════════════════════════════════════════════════════════════════════════
ax.text(FW/2, FH - 0.28,
        "VERA  —  Vision · Experience · Reasoning · Action",
        ha="center", va="center", fontsize=15, fontweight="bold", color=C_DARK, zorder=5)
ax.text(FW/2, FH - 0.72,
        "V = Vision   ·   E = Experience (outcome token, Stream 3b)   ·   "
        "R = Reasoning (LLaMA fusion backbone)   ·   A = Action",
        ha="center", va="center", fontsize=8.2, color="#546E7A", zorder=5)

# ════════════════════════════════════════════════════════════════════════════════
#  COLUMN HEADERS
# ════════════════════════════════════════════════════════════════════════════════
COL_HDRS = [
    "Encode\n(Frozen CLIP)",
    "Gate / Verbalize",
    "Project → 256-dim",
    "Token Sequence\n(+ViLT type embed)",
    "LLaMA Fusion\n(×6 layers)",
    "Dual Action Head",
]
for i in range(1, 6):
    divider(CX[i])
ax.plot([CX[0], CX[6]], [TOP_Y - 0.16, TOP_Y - 0.16], color=C_GRID, lw=0.8, zorder=1)
for i, h in enumerate(COL_HDRS):
    ax.text(MID[i], TOP_Y - 0.05, h,
            ha="center", va="center", fontsize=8.0, fontweight="bold",
            color=C_DARK, linespacing=1.25, zorder=4)

# ── Stream labels (left margin) ────────────────────────────────────────────────
for lbl, ry, col in zip(STREAM_LABELS, ROW_Y, STREAM_COLORS):
    ax.text(CX[0] - 0.10, ry, lbl,
            ha="right", va="center", fontsize=6.8,
            color=col, fontweight="bold", linespacing=1.3, zorder=4)

# ════════════════════════════════════════════════════════════════════════════════
#  DASHED FEEDBACK LOOP BOX  (Streams 3a, 3b, 4 — rows 2,3,4)
# ════════════════════════════════════════════════════════════════════════════════
fb_top  = ROW_Y[2] + ROW_H/2 + 0.45
fb_bot  = ROW_Y[4] - ROW_H/2 - 0.52
fb_left = CX[0] + 0.08
fb_right= CX[3] - 0.08
ax.add_patch(FancyBboxPatch(
    (fb_left, fb_bot), fb_right - fb_left, fb_top - fb_bot,
    boxstyle="round,pad=0.10,rounding_size=0.18",
    fc="none", ec=C_LOOP, lw=2.1, ls="--", alpha=0.85, zorder=2))
ax.text(fb_left + 0.12, fb_top + 0.16,
        "Closed-Loop Feedback  (step t  →  inputs at step t+1)",
        ha="left", va="center", fontsize=7.0, color=C_LOOP, fontweight="bold", zorder=5)

# ════════════════════════════════════════════════════════════════════════════════
#  COL 0 — ENCODE
# ════════════════════════════════════════════════════════════════════════════════
EW = CX[1] - CX[0] - 0.32
EH = 0.76

# Frozen CLIP shared panel (rows 0–3)
clip_top = ROW_Y[0] + EH/2 + 0.22
clip_bot = ROW_Y[3] - EH/2 - 0.22
ax.add_patch(FancyBboxPatch(
    (CX[0] + 0.14, clip_bot), CX[1] - CX[0] - 0.28, clip_top - clip_bot,
    boxstyle="round,pad=0.06", fc="#ECEFF1", ec=C_CLIP,
    lw=1.4, ls="--", alpha=0.80, zorder=2))
ax.text(MID[0], (clip_top + clip_bot)/2, "Frozen\nCLIP\nViT-B/32",
        ha="center", va="center", fontsize=5.4, color=C_CLIP,
        fontweight="bold", linespacing=1.2, rotation=90, zorder=3)

box(MID[0], ROW_Y[0], EW, EH, C_CLIP, "CLIP Image\nEncoder", fs=7.0)
box(MID[0], ROW_Y[1], EW, EH, C_CLIP, "CLIP Text\nEncoder",  fs=7.0)
box(MID[0], ROW_Y[2], EW, EH, C_CLIP, "CLIP Text\nEncoder",  fs=7.0)
box(MID[0], ROW_Y[3], EW, EH, C_CLIP, "CLIP Text\nEncoder",  fs=7.0)

# History encoder — 3-signal: Embed + gate·Linear(vec) + Linear(r)
box(MID[0], ROW_Y[4], EW, EH + 0.30, C_HIS,
    "3 signals:\nEmbed(idx)\n+ σ·Lin(vec)\n+ Lin(r)", fs=6.0)

# ════════════════════════════════════════════════════════════════════════════════
#  COL 1 — GATE / VERBALIZE
# ════════════════════════════════════════════════════════════════════════════════
GW = CX[2] - CX[1] - 0.28
GH = 0.76

# Vision & Instruction — pass-through (no gate/verb step)
box(MID[1], ROW_Y[0], GW, GH, C_PASS,
    "→ pass-through\n(3 frames, 224²)", fs=6.0, tc=C_DARK, ec=C_GRID, lw=0.8, zo=2)
box(MID[1], ROW_Y[1], GW, GH, C_PASS,
    "→ pass-through\n(77 BPE tokens)", fs=6.0, tc=C_DARK, ec=C_GRID, lw=0.8, zo=2)

# Stream 3a — reward gate
box(MID[1], ROW_Y[2], GW, GH + 0.18, C_GATE,
    "Reward Gate\nσ(MLP(r)) ∈ (0,1)", fs=6.6, bold=True)
ax.text(MID[1], ROW_Y[2] - 0.62,
        "attenuates failed actions",
        ha="center", va="center", fontsize=5.4, color=C_GATE,
        fontstyle="italic", zorder=5)

# Stream 3b — verbalize
box(MID[1], ROW_Y[3], GW, GH + 0.18, C_VERB,
    "verbalize_consequence\n(r, Δd) → string", fs=6.2, bold=True)
ax.text(MID[1], ROW_Y[3] - 0.62,
        "16 semantically distinct strings",
        ha="center", va="center", fontsize=5.4, color=C_VERB,
        fontstyle="italic", zorder=5)
badge(MID[1] + GW/2 - 0.22, ROW_Y[3] + GH/2 + 0.22)

# Stream 4 — fusion linear + SinPE + TemporalTF
box(MID[1], ROW_Y[4], GW, GH + 0.60, "#6A1B9A",
    "fusion Linear(3d→d)\n+ RMSNorm + SiLU\n+ SinPE\n+ TemporalTF (×2)", fs=5.8)
ax.text(MID[1], ROW_Y[4] - 0.74,
        "2-layer LLaMA sub-TF",
        ha="center", va="center", fontsize=5.4, color="#6A1B9A",
        fontstyle="italic", zorder=5)

# Encode → Gate/Verb arrows
for ry, col in zip(ROW_Y, STREAM_COLORS):
    arr(MID[0] + EW/2 + 0.04, ry, MID[1] - GW/2 - 0.04, col=col)

# ════════════════════════════════════════════════════════════════════════════════
#  COL 2 — PROJECT → 256-dim
# ════════════════════════════════════════════════════════════════════════════════
PW = CX[3] - CX[2] - 0.26
PH = 0.76

proj_rows = [
    ("vis_proj\nLinear(512→256)\n+ RMSNorm",      C_VIS),
    ("lang_proj\nLinear(512→256)\n+ RMSNorm",     C_INS),
    ("act_proj\nLinear(512→256)\n+ RMSNorm",      C_ACT),
    ("exp_proj\nLinear(512→256)\n+ RMSNorm",      C_EXP),
    ("Already 256-dim\n(output of fusion\n+ SinPE + TempTF)", C_HIS),
]
for (lbl, col), ry in zip(proj_rows, ROW_Y):
    box(MID[2], ry, PW, PH + 0.18, col, lbl, fs=6.2)

badge(MID[2] + PW/2 - 0.22, ROW_Y[3] + PH/2 + 0.28)

# Gate/Verb → Project arrows
for ry, col in zip(ROW_Y, STREAM_COLORS):
    arr(MID[1] + GW/2 + 0.04, ry, MID[2] - PW/2 - 0.04, col=col)

# ════════════════════════════════════════════════════════════════════════════════
#  COL 3 — TOKEN SEQUENCE
# ════════════════════════════════════════════════════════════════════════════════
TW = CX[4] - CX[3] - 0.24
TH = 0.78

tok_rows = [
    ("V₁  V₂  V₃\n(type=2)",     C_VIS),
    ("L_instr\n(type=0)",         C_INS),
    ("E_act\n(type=1)",           C_ACT),
    ("E_exp\n(type=4)",           C_EXP),
    ("H₁ H₂ H₃ H₄\n(type=3)",    C_HIS),
]
for (lbl, col), ry in zip(tok_rows, ROW_Y):
    box(MID[3], ry, TW, TH, col, lbl, fs=7.2)

box(MID[3], CLS_Y, TW, TH, C_FUSE, "CLS\n(type=0)", fs=7.2, bold=True)

ax.text(MID[3], CLS_Y - 0.66,
        "[ L_instr | E_act | E_exp | V₁V₂V₃ | H₁..H₄ | CLS ]",
        ha="center", va="center", fontsize=5.2, color="#546E7A",
        fontstyle="italic", zorder=4)
ax.text(MID[3], CLS_Y - 0.90,
        "+ ViLT modality-type embeddings added to each token",
        ha="center", va="center", fontsize=5.0, color="#546E7A",
        fontstyle="italic", zorder=4)

# Project → Token arrows
for ry, col in zip(ROW_Y, STREAM_COLORS):
    arr(MID[2] + PW/2 + 0.04, ry, MID[3] - TW/2 - 0.04, col=col)

# ════════════════════════════════════════════════════════════════════════════════
#  COL 4 — LLaMA FUSION TRANSFORMER (×6)
# ════════════════════════════════════════════════════════════════════════════════
# Token → Transformer entry arrows
for ry, col in zip(ROW_Y + [CLS_Y], STREAM_COLORS + [C_FUSE]):
    arr(MID[3] + TW/2 + 0.04, ry, CX[4] + 0.14, ry, col=col, lw=1.2)

t_top = ROW_Y[0] + 0.60
t_bot = CLS_Y   - 0.60
t_w   = CX[5] - CX[4] - 0.28
t_cx  = (CX[4] + CX[5]) / 2

ax.add_patch(FancyBboxPatch(
    (CX[4] + 0.14, t_bot), t_w, t_top - t_bot,
    boxstyle="round,pad=0.12,rounding_size=0.20",
    fc=C_FUSE, ec=C_DARK, lw=1.6, alpha=0.93, zorder=3))

layers = [
    ("RMSNorm",                    "#00796B"),
    ("Multi-Head Attention\n+ RoPE  (d=256, h=8)", "#00695C"),
    ("⊕  residual",                "#004D40"),
    ("RMSNorm",                    "#00796B"),
    ("SwiGLU FFN  (d_ff = 1024)",  "#00695C"),
    ("⊕  residual",                "#004D40"),
    ("Dropout  (p=0.1)",           "#003D33"),
]
stk_top = t_top - 0.24
stk_bot = t_bot + 0.24
lh = (stk_top - stk_bot) / len(layers)
iw = t_w - 0.44
for i, (name, lc) in enumerate(layers):
    lcy = stk_top - lh * (i + 0.5)
    box(t_cx, lcy, iw, lh * 0.82, lc, name,
        fs=5.6 if "\n" in name else 6.1, ec="#003D33", lw=0.8)

ax.text(CX[5] - 0.22, (t_top + t_bot)/2, "×6",
        ha="center", va="center", fontsize=14,
        color="white", fontweight="bold", zorder=6)
ax.text(t_cx, t_bot - 0.24,
        "Causal mask — CLS (pos −1) attends to all 10 preceding tokens",
        ha="center", va="center", fontsize=5.4,
        color="#546E7A", fontstyle="italic", zorder=4)

# ════════════════════════════════════════════════════════════════════════════════
#  COL 5 — DUAL ACTION HEAD
# ════════════════════════════════════════════════════════════════════════════════
HX0 = CX[5] + 0.18
HX1 = CX[6] - 0.20
H_MID = (HX0 + HX1) / 2

disc_y = (ROW_Y[0] + ROW_Y[1]) / 2 + 0.05
reg_y  = (ROW_Y[3] + CLS_Y)    / 2 + 0.08
fork_y = (disc_y + reg_y) / 2
split_x = HX0 - 0.10

# Arrow from LLaMA CLS → fork
arr(CX[5] - 0.14, fork_y, split_x, col=C_FUSE, lw=2.2)
for branch_y in [disc_y, reg_y]:
    ax.plot([split_x, split_x, split_x + 0.14],
            [fork_y, branch_y, branch_y],
            color=C_DARK, lw=1.8, zorder=5)

# ── Branch 1: Discrete (3-layer MLP matching code exactly) ─────────────────────
# RMSNorm → Linear(d→d) + SiLU + Drop → Linear(d→d/2) + SiLU + Drop → Linear(d/2→N) → argmax
disc_steps = [
    "RMSNorm",
    "Linear(d→d)\nSiLU + Drop",
    "Linear(d→d/2)\nSiLU + Drop",
    "Linear\n(d/2→N)",
    "argmax\n→ aₜ",
]
disc_n = len(disc_steps)
disc_total = HX1 - (split_x + 0.14)
disc_bw = disc_total / disc_n
disc_bh = 0.64

ax.text(H_MID, disc_y + 0.80,
        "Branch 1 — Discrete Action Head  (3-layer MLP)",
        ha="center", va="center", fontsize=7.2, color=C_DISC, fontweight="bold", zorder=5)

for k, lbl in enumerate(disc_steps):
    bx = split_x + 0.14 + disc_bw * (k + 0.5)
    box(bx, disc_y, disc_bw - 0.10, disc_bh, C_DISC, lbl, fs=5.6, ec=C_DARK, lw=0.8)
    if k < disc_n - 1:
        arr(bx + disc_bw/2 - 0.03, disc_y, bx + disc_bw/2 + 0.07, col=C_DISC, lw=1.0)

ax.text(H_MID, disc_y - 0.64,
        "L_BC = cross-entropy + label smoothing (ε=0.05)",
        ha="center", va="center", fontsize=5.5, color=C_DISC,
        fontstyle="italic", zorder=4)

# ── Branch 2: Regression (2-layer MLP matching code exactly) ───────────────────
# RMSNorm → Linear(d→d/2) + SiLU + Drop → Linear(d/2→action_dim) → Tanh
reg_steps = [
    "RMSNorm",
    "Linear(d→d/2)\nSiLU + Drop",
    "Linear\n(d/2→action_dim)",
    "Tanh",
    "v̂ₜ ∈\n[-1,1]^d",
]
reg_n = len(reg_steps)
reg_total = HX1 - (split_x + 0.14)
reg_bw = reg_total / reg_n
reg_bh = 0.64

ax.text(H_MID, reg_y + 0.80,
        "Branch 2 — Continuous Regression Head  (2-layer MLP)",
        ha="center", va="center", fontsize=7.2, color=C_REG, fontweight="bold", zorder=5)
badge(H_MID + 2.10, reg_y + 0.80)

for k, lbl in enumerate(reg_steps):
    bx = split_x + 0.14 + reg_bw * (k + 0.5)
    box(bx, reg_y, reg_bw - 0.10, reg_bh, C_REG, lbl, fs=5.6, ec=C_DARK, lw=0.8)
    if k < reg_n - 1:
        arr(bx + reg_bw/2 - 0.03, reg_y, bx + reg_bw/2 + 0.07, col=C_REG, lw=1.0)

ax.text(H_MID, reg_y - 0.64,
        "L_reg = MSE vs expert action vector  (inactive when target_vec=None)",
        ha="center", va="center", fontsize=5.5, color=C_REG,
        fontstyle="italic", zorder=4)

# Divider between branches
ax.plot([HX0, HX1], [(disc_y + reg_y)/2]*2,
        color=C_GRID, lw=0.8, ls="--", zorder=2)

# ════════════════════════════════════════════════════════════════════════════════
#  FEEDBACK ARROWS (right edge → left inputs of streams 3a, 3b, 4)
# ════════════════════════════════════════════════════════════════════════════════
fb_x = HX1 + 0.32

# Stubs from each branch's last box
for branch_y in [disc_y, reg_y]:
    last_cx = split_x + 0.14 + disc_total - disc_bw/2 + 0.05
    ax.plot([last_cx, fb_x], [branch_y, branch_y], color=C_LOOP, lw=1.9, zorder=5)

# Vertical track
ax.plot([fb_x, fb_x], [disc_y, ROW_Y[4]], color=C_LOOP, lw=2.1, zorder=5)

# Arrows left into streams 3a, 3b, 4 at encode column
for ry, col in zip(ROW_Y[2:], STREAM_COLORS[2:]):
    ax.annotate("", xy=(CX[0] + EW/2 + 0.12, ry), xytext=(fb_x, ry),
                arrowprops=dict(arrowstyle="-|>", color=C_LOOP,
                                lw=1.6, mutation_scale=12), zorder=6)

ax.text(fb_x + 0.13, (disc_y + ROW_Y[4])/2, "step\nt+1",
        ha="left", va="center", fontsize=6.5,
        color=C_LOOP, fontweight="bold", rotation=90, zorder=5)

# ════════════════════════════════════════════════════════════════════════════════
#  LEGEND (bottom left)
# ════════════════════════════════════════════════════════════════════════════════
leg_items = [
    (C_VIS,  "Vision stream"),
    (C_INS,  "Instruction stream"),
    (C_ACT,  "Action Narration (E_act, type=1)"),
    (C_EXP,  "Experience / E_exp (type=4)  ← E in VERA"),
    (C_HIS,  "Action History (type=3)"),
    (C_FUSE, "LLaMA Reasoning Transformer  ← R in VERA"),
    (C_LOOP, "Closed-loop feedback"),
]
lx0, ly0 = 0.35, 3.20
for i, (col, lbl) in enumerate(leg_items):
    lx = lx0 + (i % 4) * 6.20
    ly = ly0 - (i // 4) * 0.54
    ax.add_patch(mpatches.Rectangle((lx, ly - 0.14), 0.28, 0.28, fc=col, ec="none", zorder=5))
    ax.text(lx + 0.38, ly, lbl, ha="left", va="center", fontsize=6.0, color=C_DARK, zorder=5)

# ════════════════════════════════════════════════════════════════════════════════
#  ACRONYM CALLOUT (bottom right)
# ════════════════════════════════════════════════════════════════════════════════
cx0, cy0 = FW - 0.35, 3.60
ax.text(cx0, cy0 + 0.44, "VERA Acronym Map",
        ha="right", va="center", fontsize=7.8, color=C_DARK, fontweight="bold", zorder=5)
for i, (letter, col, desc) in enumerate([
    ("V", C_VIS,  " = Vision  (Stream 1)"),
    ("E", C_EXP,  " = Experience  (Stream 3b, E_exp, type=4)"),
    ("R", C_FUSE, " = Reasoning  (LLaMA fusion backbone)"),
    ("A", C_DISC, " = Action  (dual head: discrete + regression)"),
]):
    ly = cy0 - i * 0.44
    ax.text(cx0 - 1.30, ly, letter, ha="left", va="center",
            fontsize=12, color=col, fontweight="bold", zorder=5)
    ax.text(cx0 - 1.05, ly, desc, ha="left", va="center",
            fontsize=6.5, color=C_DARK, zorder=5)

# ════════════════════════════════════════════════════════════════════════════════
#  SAVE
# ════════════════════════════════════════════════════════════════════════════════
out = "docs/VERA.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
print(f"✓  Saved: {out}")
plt.close(fig)
