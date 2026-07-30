"""
Regenerate loss.png and accuracy.png.
Exact port of the original Canvas/JS artifacts, new aspect ratio (narrower x, taller y).
"""

import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline

LOG  = pathlib.Path.home() / "Downloads" / "sft_vera_log.json"
OUT  = pathlib.Path(__file__).parent
BEST = 43

# ── exact colours from the original artifacts ─────────────────────────────────
TRAIN   = '#1456A0'
VAL     = '#BA4E18'
BEST_C  = '#0A8056'
GRID_C  = '#e8ecf2'
BORDER  = '#c8d0dc'
PLOT_BG = '#fafbfc'
INK2    = '#4a5568'
INK3    = '#8090a0'
WARMUP_C= (99/255, 51/255, 168/255, 0.07)   # rgba(99,51,168,.07)

# fill-under alpha: '18' hex = 24/255 ≈ 0.094
TRAIN_FILL = TRAIN + '18'   # not valid hex for mpl — use tuple below
VAL_FILL   = VAL   + '18'

def hex_alpha(hex6, alpha):
    r = int(hex6[1:3],16)/255
    g = int(hex6[3:5],16)/255
    b = int(hex6[5:7],16)/255
    return (r, g, b, alpha)

TRAIN_A = hex_alpha(TRAIN, 24/255)
VAL_A   = hex_alpha(VAL,   24/255)

# ── new aspect ratio ──────────────────────────────────────────────────────────
# Original: 900×420 px → ~6.0×2.8 in.  New: narrower x, taller y → ~4.5×4.8 in
FW, FH = 6.0, 5.5

def load():
    with open(LOG) as f:
        rows = json.load(f)
    ep = np.array([r["epoch"]      for r in rows], dtype=float)
    tl = np.array([r["train_loss"] for r in rows])
    vl = np.array([r["val_loss"]   for r in rows])
    ta = np.array([r["train_acc"]  for r in rows])
    va = np.array([r["val_acc"]    for r in rows])
    return ep, tl, vl, ta, va

def smooth_line(ep, y):
    """Catmull-Rom-like smoothing matching the JS bezier tension=0.15."""
    ep_fine = np.linspace(ep[0], ep[-1], 400)
    spl = make_interp_spline(ep, y, k=3)
    return ep_fine, spl(ep_fine)

def base_ax(ax):
    ax.set_facecolor(PLOT_BG)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
        sp.set_linewidth(0.8)
    ax.tick_params(colors=INK3, labelsize=8.5, length=3, width=0.8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.yaxis.grid(True, color=GRID_C, linewidth=0.8, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

def mark43(ax):
    fig = ax.get_figure(); fig.canvas.draw()
    for lbl in ax.get_xticklabels():
        if lbl.get_text().strip() == '43':
            lbl.set_color(BEST_C); lbl.set_fontweight('bold')


# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────
def plot_loss(ep, tl, vl):
    fig, ax = plt.subplots(figsize=(FW, FH), dpi=150)
    fig.patch.set_facecolor('white')
    base_ax(ax)

    ep_s, tl_s = smooth_line(ep, tl)
    _,    vl_s = smooth_line(ep, vl)
    y_floor = min(tl_s.min(), vl_s.min()) - 0.003

    # warmup shading
    ax.axvspan(1, 5, color=WARMUP_C, zorder=1)
    ax.text(1.3, tl[0]+0.0005, 'warmup', fontsize=7.5,
            color='#6333A8', alpha=0.7, va='bottom', zorder=6)

    # fill under each curve independently (matching JS: fillUnder TRAIN then VAL)
    ax.fill_between(ep_s, y_floor, tl_s, color=TRAIN_A, zorder=2)
    ax.fill_between(ep_s, y_floor, vl_s, color=VAL_A,   zorder=2)

    # lines
    ax.plot(ep_s, tl_s, color=TRAIN, lw=1.8, zorder=4, label='Train loss')
    ax.plot(ep_s, vl_s, color=VAL,   lw=1.8, zorder=4, label='Val loss')

    # best checkpoint
    ax.axvline(BEST, color=BEST_C, lw=1.4, linestyle='--', zorder=5,
               label=f'Best ckpt (ep {BEST})', alpha=0.8)

    # x-ticks with 43
    ax.set_xticks([1, 10, 20, 30, 40, 43, 50, 60])
    ax.set_xlim(1, 60)
    ax.set_ylim(y_floor, tl[0]+0.005)

    ax.set_yticks([.370, .380, .390, .400, .410])
    ax.yaxis.set_major_formatter(lambda x, _: f'{x:.3f}')

    ax.set_xlabel('Epoch', fontsize=9.5)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=9.5)
    ax.set_title('Cross-Entropy Loss — TERM SFT (60 epochs)',
                 fontsize=10.5, fontweight='bold', pad=7, loc='left',
                 x=0.0, color='#1a2a3a')

    legend_lines = [
        Line2D([0],[0], color=TRAIN, lw=2, label='Train loss'),
        Line2D([0],[0], color=VAL,   lw=2, label='Val loss'),
        Line2D([0],[0], color=BEST_C,lw=1.5, linestyle='--', label=f'Best ckpt (ep {BEST})'),
    ]
    ax.legend(handles=legend_lines, fontsize=8, loc='upper right',
              framealpha=0.92, edgecolor=BORDER, handlelength=1.8)

    mark43(ax)
    fig.tight_layout(pad=0.7)
    out = OUT / 'loss.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY
# ─────────────────────────────────────────────────────────────────────────────
def plot_accuracy(ep, ta, va):
    fig, ax = plt.subplots(figsize=(FW, FH), dpi=150)
    fig.patch.set_facecolor('white')
    base_ax(ax)

    ta_pct = ta * 100
    va_pct = va * 100
    ep_s, ta_s = smooth_line(ep, ta_pct)
    _,    va_s = smooth_line(ep, va_pct)
    y_floor = min(ta_s.min(), va_s.min()) - 0.05

    # warmup shading
    ax.axvspan(1, 5, color=WARMUP_C, zorder=1)

    # fill under each curve independently
    ax.fill_between(ep_s, y_floor, ta_s, color=TRAIN_A, zorder=2)
    ax.fill_between(ep_s, y_floor, va_s, color=VAL_A,   zorder=2)

    # lines with small dots matching original
    ax.plot(ep_s, ta_s, color=TRAIN, lw=1.8, zorder=4)
    ax.plot(ep_s, va_s, color=VAL,   lw=1.8, zorder=4)
    # dots at actual data points
    ax.scatter(ep, ta_pct, color=TRAIN, s=6, zorder=5)
    ax.scatter(ep, va_pct, color=VAL,   s=6, zorder=5)

    # best checkpoint
    ax.axvline(BEST, color=BEST_C, lw=1.4, linestyle='--', zorder=5, alpha=0.8)

    ax.set_xticks([1, 10, 20, 30, 40, 43, 50, 60])
    ax.set_xlim(1, 60)
    ax.set_ylim(y_floor, max(ta_s.max(), va_s.max()) + 0.05)
    ax.yaxis.set_major_formatter(lambda x, _: f'{x:.2f}%')

    ax.set_xlabel('Epoch', fontsize=9.5)
    ax.set_ylabel('Accuracy', fontsize=9.5)
    ax.set_title('Classification Accuracy — TERM SFT (60 epochs)',
                 fontsize=10.5, fontweight='bold', pad=7, loc='left',
                 x=0.0, color='#1a2a3a')

    # horizontal legend above plot
    legend_items = [
        Line2D([0],[0], color=TRAIN, lw=2, marker='o', markersize=3, label='Train acc'),
        Line2D([0],[0], color=VAL,   lw=2, marker='o', markersize=3, label='Val acc'),
        Line2D([0],[0], color=BEST_C,lw=1.5, linestyle='--', label=f'Best checkpoint (ep {BEST})'),
        mpatches.Patch(facecolor=(99/255,51/255,168/255,0.12),
                       edgecolor='#6333A8', lw=0.8, label='Warmup (ep 1–5)'),
    ]
    ax.legend(handles=legend_items, loc='upper center',
              bbox_to_anchor=(0.5, 1.14), ncol=4,
              fontsize=7.5, framealpha=0.92, edgecolor=BORDER, handlelength=1.6)

    mark43(ax)
    fig.tight_layout(pad=0.7)
    out = OUT / 'accuracy.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    ep, tl, vl, ta, va = load()
    plot_loss(ep, tl, vl)
    plot_accuracy(ep, ta, va)
