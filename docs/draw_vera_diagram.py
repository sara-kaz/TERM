"""
VERA Architecture Diagram — clean v2
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
C_VIS   = "#2471A3"
C_INST  = "#1E8449"
C_ACT   = "#D35400"
C_EXP   = "#922B21"
C_HIST  = "#6C3483"
C_CLIP  = "#2C3E50"
C_FUSE  = "#17202A"
C_HEAD  = "#154360"
C_CLS   = "#1A5276"
C_FBK   = "#626567"
C_CHUNK = "#1D8348"
C_REG   = "#1F618D"
C_GOLD  = "#B7950B"
C_WHITE = "#FFFFFF"
C_LIGHT = "#F2F3F4"
C_BIDIR = "#C0392B"

SM  = 6.8
MD  = 8.0
LG  = 9.5

def box(ax, x, y, w, h, fc, text, fs=MD, tc=C_WHITE,
        bold=False, rad=0.15, alpha=1.0, ls="-", ec=None, zorder=3):
    ec = ec or fc
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.04,rounding_size={rad}",
                       fc=fc, ec=ec, lw=1.1, alpha=alpha,
                       ls=ls, zorder=zorder)
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight="bold" if bold else "normal",
            multialignment="center", zorder=zorder+1)

def arr(ax, x0,y0,x1,y1, col="#555555", lw=1.3, rad=0.0, zorder=2):
    ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=zorder)

def txt(ax, x, y, s, fs=SM, col="#333333", ha="center", va="center",
        bold=False, italic=False, zorder=5):
    style = "italic" if italic else "normal"
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=col,
            weight="bold" if bold else "normal",
            style=style, zorder=zorder)

# ── Figure ────────────────────────────────────────────────────────────────────
FW, FH = 26, 13
fig, ax = plt.subplots(figsize=(FW, FH), dpi=150)
ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis("off")
fig.patch.set_facecolor(C_LIGHT); ax.set_facecolor(C_LIGHT)

# Title
txt(ax, FW/2, 12.70,
    "VERA — Vision · Experience · Reasoning · Action",
    fs=15, col="#1A1A2E", bold=True)
txt(ax, FW/2, 12.22,
    "5-stream closed-loop robot policy  ·  Bidirectional LLaMA fusion  ·  K=4 action chunking  ·  CoT-lite expand-compress head",
    fs=SM+0.5, col="#555555", italic=True)

# ── Section backgrounds ───────────────────────────────────────────────────────
def sbg(ax, x, w, label, col="#AAAAAA"):
    r = FancyBboxPatch((x, 0.4), w, 11.5,
                       boxstyle="round,pad=0.1",
                       fc=col, ec="#BBBBBB", lw=0.6,
                       alpha=0.13, zorder=0)
    ax.add_patch(r)
    txt(ax, x+w/2, 11.70, label, fs=SM-0.5, col="#777777", italic=True)

sbg(ax,  0.15,  2.6,  "Inputs")
sbg(ax,  2.95,  3.8,  "Encode & Gate")
sbg(ax,  6.95,  2.3,  "Token Sequence\n(+ ViLT embeddings)")
sbg(ax,  9.45,  3.0,  "LLaMA Fusion ×6\n(Bidirectional)")
sbg(ax, 12.65, 10.0,  "Dual Action Head + Alignment Loss")

# ── Stream row y-centres ──────────────────────────────────────────────────────
Y1 = 9.60   # Vision
Y2 = 7.70   # Instruction
Y3 = 5.80   # Stream 3a
Y4 = 3.90   # Stream 3b
Y5 = 2.00   # History
BH = 0.62

# ── Stream labels ─────────────────────────────────────────────────────────────
for yc, lab, col in [
    (Y1, "Stream 1\nVision",        C_VIS),
    (Y2, "Stream 2\nInstruction",   C_INST),
    (Y3, "Stream 3a\nAct. Narr.",   C_ACT),
    (Y4, "Stream 3b\nExperience",   C_EXP),
    (Y5, "Stream 4\nHistory",       C_HIST),
]:
    txt(ax, 0.05, yc, lab, fs=SM-0.5, col=col, ha="left", bold=True)

# ── Column 1: Inputs  x=0.85-2.35 ────────────────────────────────────────────
IX, IW = 0.85, 1.90
for yc, col, lab in [
    (Y1, C_VIS,  "3 RGB Frames\n(224×224)"),
    (Y2, C_INST, "Task Instruction\n(natural language)"),
    (Y3, C_ACT,  "Prev Action  aₜ₋₁"),
    (Y4, C_EXP,  "Reward rₜ₋₁\n& Δdist"),
    (Y5, C_HIST, "History Window\n{(aᵢ, vᵢ, rᵢ)} × H=4"),
]:
    box(ax, IX, yc-BH/2, IW, BH, col, lab, fs=SM)

# ── Column 2: Encoders  x=2.95-6.55 ─────────────────────────────────────────
E1X, E1W = 2.95, 1.65   # encoder 1
E2X, E2W = 4.85, 1.95   # projection

def stream_encode(ax, yc, enc_txt, proj_txt, ec, pc, ih=BH):
    box(ax, E1X, yc-ih/2, E1W, ih, C_CLIP, enc_txt,  fs=SM-0.5)
    box(ax, E2X, yc-ih/2, E2W, ih, pc,     proj_txt, fs=SM-0.5)
    arr(ax, IX+IW, yc, E1X, yc, col=ec)
    arr(ax, E1X+E1W, yc, E2X, yc, col=ec)

stream_encode(ax, Y1,
    "CLIP ViT-B/32\n(image encoder)",
    "vis_proj\nRMSNorm(W·e)",
    C_VIS, C_VIS)

stream_encode(ax, Y2,
    "CLIP Text\n(text encoder)",
    "lang_proj\nRMSNorm(W·e)",
    C_INST, C_INST)

# Stream 3a: two sub-boxes stacked
G3Y = Y3+0.26; C3Y = Y3-0.26
box(ax, E1X, G3Y-0.23, E1W, 0.46, C_ACT,
    "Reward Gate\nσ(MLP(r̃))·vocab[aₜ₋₁]", fs=SM-1.0)
box(ax, E1X, C3Y-0.23, E1W, 0.46, C_CLIP,
    "CLIP Text\n(frozen)", fs=SM-1.0)
arr(ax, IX+IW,     Y3,   E1X, G3Y, col=C_ACT)
arr(ax, E1X+E1W*0.5, G3Y-0.23, E1X+E1W*0.5, C3Y+0.23, col=C_ACT)
box(ax, E2X, Y3-BH/2, E2W, BH, C_ACT, "act_proj\nRMSNorm(W·e)", fs=SM-0.5)
arr(ax, E1X+E1W, C3Y, E2X, Y3, col=C_ACT)

# Stream 3b
G4Y = Y4+0.26; C4Y = Y4-0.26
box(ax, E1X, G4Y-0.23, E1W, 0.46, C_EXP,
    "verbalize_consequence\n(r, Δd) → string", fs=SM-1.0)
box(ax, E1X, C4Y-0.23, E1W, 0.46, C_CLIP,
    "CLIP Text\n(frozen)", fs=SM-1.0)
arr(ax, IX+IW,     Y4,   E1X, G4Y, col=C_EXP)
arr(ax, E1X+E1W*0.5, G4Y-0.23, E1X+E1W*0.5, C4Y+0.23, col=C_EXP)
box(ax, E2X, Y4-BH/2, E2W, BH, C_EXP, "con_proj\nRMSNorm(W·e)", fs=SM-0.5)
arr(ax, E1X+E1W, C4Y, E2X, Y4, col=C_EXP)

# History
box(ax, E1X, Y5-0.36, E1W, 0.72, C_HIST,
    "HistoryEncoder\n(discrete+vec+reward)", fs=SM-1.0)
box(ax, E2X, Y5-0.36, E2W, 0.72, C_HIST,
    "TemporalTF (2L)\n(causal — internal)", fs=SM-1.0)
arr(ax, IX+IW, Y5, E1X, Y5, col=C_HIST)
arr(ax, E1X+E1W, Y5, E2X, Y5, col=C_HIST)

# ── Column 3: Token Sequence  x=6.95-9.25 ────────────────────────────────────
TX, TW, TH = 6.95, 2.10, 0.59

token_data = [
    # (y_centre, color, label)
    (10.55, C_INST, "L_instr   (type 0)"),
    (9.92,  C_ACT,  "E_act      (type 1)"),
    (9.29,  C_EXP,  "E_exp      (type 4)"),
    (8.66,  C_VIS,  "V₁          (type 2)"),
    (8.03,  C_VIS,  "V₂          (type 2)"),
    (7.40,  C_VIS,  "V₃          (type 2)"),
    (6.77,  C_HIST, "H₁          (type 3)"),
    (6.14,  C_HIST, "H₂          (type 3)"),
    (5.51,  C_HIST, "H₃          (type 3)"),
    (4.88,  C_HIST, "H₄          (type 3)"),
    (4.25,  C_CLS,  "CLS         (type 0)"),
]
for yc, col, lab in token_data:
    box(ax, TX, yc-TH/2, TW, TH, col, lab, fs=SM-0.3, rad=0.10)

# Arrows: projections → tokens
arr(ax, E2X+E2W, Y2, TX, 10.55, col=C_INST)   # instr → L_instr
arr(ax, E2X+E2W, Y3, TX,  9.92, col=C_ACT)    # act   → E_act
arr(ax, E2X+E2W, Y4, TX,  9.29, col=C_EXP)    # exp   → E_exp
for yy in [8.66, 8.03, 7.40]:                  # vis frames
    arr(ax, E2X+E2W, Y1, TX, yy, col=C_VIS)
for yy in [6.77, 6.14, 5.51, 4.88]:            # history
    arr(ax, E2X+E2W, Y5, TX, yy, col=C_HIST)

# ── Column 4: LLaMA Fusion  x=9.45-12.25 ────────────────────────────────────
FX, FW_box = 9.45, 2.80
F_BOT = 4.25 - TH/2 - 0.20
F_TOP = 10.55 + TH/2 + 0.20
F_H   = F_TOP - F_BOT

box(ax, FX, F_BOT, FW_box, F_H, C_FUSE,
    "LLaMA Fusion Transformer\n\n"
    "6 layers · 8 heads · d = 256\n\n"
    "RMSNorm · RoPE · SwiGLU\n\n"
    "× 6",
    fs=MD, bold=False)

# Bidirectional badge
BD_W, BD_H = FW_box - 0.20, 0.80
box(ax, FX+0.10, F_TOP - BD_H - 0.12, BD_W, BD_H,
    C_BIDIR, "BIDIRECTIONAL\n(no causal mask)",
    fs=SM, bold=True)

# wide arrow tokens → fusion
arr(ax, TX+TW, 7.40, FX, 7.40, col="#777777", lw=2.0)

# ── CLS output  x=12.45-13.85 ────────────────────────────────────────────────
CX, CW_, CY_, CH_ = 12.45, 1.35, 6.90, 0.68
box(ax, CX, CY_, CW_, CH_, C_CLS,
    "CLS features\n(B, d=256)", fs=SM)
arr(ax, FX+FW_box, 7.40, CX, CY_+CH_/2, col="#777777", lw=1.8)

# ── Dual Action Head ──────────────────────────────────────────────────────────
HX  = 14.05   # head start x
BW1 = 2.65    # expand-compress box width
BW2 = 2.50    # chunk output box width
BW3 = 2.00    # execute box width
BW4 = 2.65    # regression head width
BW5 = 3.20    # continuous output box

GAP = 0.28    # gap between boxes

# Branch 1 — Discrete (y-centre 9.00)
D_Y = 9.00
box(ax, HX, D_Y-0.38, BW1, 0.76, C_HEAD,
    "Expand-Compress Bottleneck\n(CoT-lite: 256→512→256)", fs=SM-0.5)
arr(ax, HX+BW1, D_Y, HX+BW1+GAP, D_Y, col=C_HEAD)

box(ax, HX+BW1+GAP, D_Y-0.38, BW2, 0.76, C_HEAD,
    "K=4 Action Chunk\naₜ  aₜ₊₁  aₜ₊₂  aₜ₊₃", fs=SM-0.5)
arr(ax, HX+BW1+GAP+BW2, D_Y, HX+BW1+GAP+BW2+GAP, D_Y, col=C_CHUNK)

box(ax, HX+BW1+GAP+BW2+GAP, D_Y-0.30, BW3, 0.60, C_CHUNK,
    "Execute  aₜ\n(argmax, step t only)", fs=SM-0.5)

# dashed box over aₜ₊₁..aₜ₊₃
dx = HX+BW1+GAP + BW2*0.38
dw = BW2*0.62
p = FancyBboxPatch((dx, D_Y-0.34), dw, 0.68,
                   boxstyle="round,pad=0.03",
                   fc="#d5f5e3", ec=C_CHUNK, lw=0.9,
                   ls="--", alpha=0.50, zorder=4)
ax.add_patch(p)
txt(ax, dx+dw/2, D_Y+0.50,
    "training targets only", fs=SM-1.5, col=C_CHUNK, italic=True)

# Branch 2 — Continuous (y-centre 5.80)
R_Y = 5.80
box(ax, HX, R_Y-0.38, BW1, 0.76, C_HEAD,
    "Regression Head\n(256→128, RMSNorm + Tanh)", fs=SM-0.5)
arr(ax, HX+BW1, R_Y, HX+BW1+GAP, R_Y, col=C_REG)

box(ax, HX+BW1+GAP, R_Y-0.30, BW5, 0.60, C_REG,
    "Continuous action  v̂ₜ ∈ (−1, 1)ᵈ\n"
    "d=2 (LT)  ·  d=4 (MetaWorld)  ·  d=7 (CALVIN)", fs=SM-0.5)

# Connector: CLS → Branch 1 and Branch 2
arr(ax, CX+CW_, CY_+CH_*0.72, HX, D_Y,   col=C_CLS, lw=1.4)
arr(ax, CX+CW_, CY_+CH_*0.28, HX, R_Y,   col=C_CLS, lw=1.4)

# ── "Dual Action Head" brace label ───────────────────────────────────────────
brace_x = HX - 0.55
brace_yb = R_Y - 0.45
brace_yt = D_Y + 0.45
p2 = FancyBboxPatch((brace_x, brace_yb), 0.18, brace_yt-brace_yb,
                    boxstyle="round,pad=0.02",
                    fc="#DDDDDD", ec="#AAAAAA", lw=0.8, zorder=3)
ax.add_patch(p2)
txt(ax, brace_x+0.09, (brace_yb+brace_yt)/2,
    "Dual\nAction\nHead", fs=SM-0.5, col="#333333",
    bold=True, va="center")

# ── InfoNCE alignment loss ────────────────────────────────────────────────────
AL_X = HX
AL_Y = 1.50
AL_W = BW1 + GAP + BW2 + GAP + BW3
AL_H = 0.90
box(ax, AL_X, AL_Y, AL_W, AL_H, C_GOLD,
    "Reward-Weighted InfoNCE Alignment Loss  (training only)\n"
    "r̃ᵢ = rᵢ/max(r) · exp(5·r̃ᵢ) weights  ·  λ_align=1.0 · (L_act+L_exp)/2  +  λ_reg=0.5·MSE",
    fs=SM-0.5, tc="#1A1A1A", rad=0.12)

ax.annotate("", xy=(AL_X+AL_W*0.3, AL_Y+AL_H),
            xytext=(FX+FW_box/2, F_BOT-0.05),
            arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=1.1,
                            ls="dashed",
                            connectionstyle="arc3,rad=-0.25"), zorder=2)
txt(ax, FX+FW_box/2+1.2, F_BOT-0.45,
    "e_instr, e_act, e_exp\n(projected, trainable)",
    fs=SM-1.5, col=C_GOLD, italic=True)

# ── FEEDBACK LOOP ─────────────────────────────────────────────────────────────
# Route cleanly below y=0.9 so nothing overlaps
EXEC_MID_X = HX+BW1+GAP+BW2+GAP + BW3/2

# aₜ → Stream 3a
ax.annotate("",
    xy=(IX+IW*0.4, Y3-BH/2-0.02),
    xytext=(EXEC_MID_X, D_Y-0.30),
    arrowprops=dict(arrowstyle="->", color=C_ACT, lw=1.2,
                    connectionstyle="arc3,rad=0.38"), zorder=2)
txt(ax, (IX+IW*0.4+EXEC_MID_X)*0.5 - 0.5, Y3-BH/2-1.05,
    "aₜ → Stream 3a (t+1)", fs=SM-1.5, col=C_ACT, italic=True)

# rₜ, Δd → Stream 3b
ax.annotate("",
    xy=(IX+IW*0.4, Y4-BH/2-0.02),
    xytext=(EXEC_MID_X+0.4, D_Y-0.38),
    arrowprops=dict(arrowstyle="->", color=C_EXP, lw=1.2,
                    connectionstyle="arc3,rad=0.48"), zorder=2)
txt(ax, (IX+IW*0.4+EXEC_MID_X)*0.5, Y4-BH/2-1.05,
    "rₜ, Δdist → Stream 3b (t+1)", fs=SM-1.5, col=C_EXP, italic=True)

# (aₜ, vₜ, rₜ) → Stream 4
ax.annotate("",
    xy=(IX+IW*0.4, Y5-0.36-0.72/2-0.02),
    xytext=(HX+BW1+GAP+BW2*0.7, R_Y-0.40),
    arrowprops=dict(arrowstyle="->", color=C_HIST, lw=1.2,
                    connectionstyle="arc3,rad=0.50"), zorder=2)
txt(ax, (IX+IW*0.4 + HX+BW1+GAP+BW2*0.7)*0.5 - 0.4, 0.60,
    "(aₜ, vₜ, rₜ) → Stream 4 (t+1)", fs=SM-1.5, col=C_HIST, italic=True)

# feedback label
box(ax, 8.0, 0.35, 4.8, 0.48, C_FBK,
    "⟲  Closed-Loop Feedback  (all arrows re-enter at step t+1)",
    fs=SM-0.5, tc=C_WHITE, alpha=0.80, rad=0.12)

# ── LEGEND  x=22.8-25.8 ──────────────────────────────────────────────────────
LX, LY0 = 22.80, 11.50
txt(ax, LX+1.2, LY0, "Legend", fs=MD, col="#222222", bold=True)
items = [
    (C_VIS,   "Stream 1: Vision"),
    (C_INST,  "Stream 2: Instruction"),
    (C_ACT,   "Stream 3a: Action Narration"),
    (C_EXP,   "Stream 3b: Experience [NEW]"),
    (C_HIST,  "Stream 4: History"),
    (C_CLIP,  "CLIP backbone (frozen)"),
    (C_FUSE,  "LLaMA Fusion (bidirectional)"),
    (C_BIDIR, "No causal mask [fix]"),
    (C_CHUNK, "Execute aₜ  (K=4 chunk)"),
    (C_REG,   "Continuous regression"),
    (C_GOLD,  "Alignment loss (training)"),
]
for i, (col, lab) in enumerate(items):
    yy = LY0 - 0.55 - i*0.48
    p = FancyBboxPatch((LX, yy-0.14), 0.32, 0.28,
                       boxstyle="round,pad=0.02",
                       fc=col, ec="none", zorder=5)
    ax.add_patch(p)
    txt(ax, LX+0.46, yy, lab, fs=SM-0.5, ha="left")

# VERA acronym
txt(ax, LX+1.2, LY0 - 0.55 - len(items)*0.48 - 0.4,
    "VERA Acronym", fs=MD, col="#222222", bold=True)
acr = [
    ("V", "Vision  (Stream 1)",                      C_VIS),
    ("E", "Experience  (Stream 3b — the E in VERA)", C_EXP),
    ("R", "Reasoning  (LLaMA fusion backbone)",       "#444444"),
    ("A", "Action  (discrete chunk + continuous)",    C_CHUNK),
]
base_y = LY0 - 0.55 - len(items)*0.48 - 0.90
for i, (let, desc, col) in enumerate(acr):
    yy = base_y - i*0.44
    txt(ax, LX+0.06, yy, f"{let} =", fs=MD, ha="left", col=col, bold=True)
    txt(ax, LX+0.55, yy, desc, fs=SM-0.5, ha="left")

# Technical notes
note_y = base_y - len(acr)*0.44 - 0.55
notes = ("Mode A: ~7.5M trainable   Mode B: +86.2M (CLIP vision)\n"
         "chunk_size K=4: train on aₜ..aₜ₊₃, execute aₜ only\n"
         "λ_align=1.0 · exp(5r̃) InfoNCE  ·  λ_reg=0.5 · MSE")
txt(ax, LX, note_y, notes, fs=SM-1.5, col="#666666", ha="left",
    italic=True)

# ── Save ──────────────────────────────────────────────────────────────────────
out = "/Users/HP/Desktop/University/VLA-Robot-Learning/docs/VERA.png"
plt.tight_layout(pad=0.3)
plt.savefig(out, dpi=150, bbox_inches="tight",
            facecolor=C_LIGHT, edgecolor="none")
print(f"Saved → {out}")
plt.close()
