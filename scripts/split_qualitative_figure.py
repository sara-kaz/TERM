"""
Split combined_lt_calvin_composite.png into 12 individual episode frames.
Outputs: docs/qual_frames/qual_r{1-4}_{start|mid|end}.png
"""
from PIL import Image
from pathlib import Path

SRC = Path(__file__).parent.parent / "docs" / "combined_lt_calvin_composite.png"
OUT = Path(__file__).parent.parent / "docs" / "qual_frames"
OUT.mkdir(exist_ok=True)

img = Image.open(SRC).convert("RGB")

PAD = 4  # pixels to trim from each edge to remove border artifacts

# Column spans for the 3 image columns (detected via brightness analysis)
COL_SPANS = {
    "start": (633 + PAD, 1103 - PAD),
    "mid":   (1244 + PAD, 1714 - PAD),
    "end":   (1855 + PAD, 2325 - PAD),
}

# Row spans for the 4 episode rows (detected via brightness analysis)
ROW_SPANS = {
    "r1": (76  + PAD, 550  - PAD),   # Language-Table ep 1
    "r2": (682 + PAD, 1155 - PAD),   # Language-Table ep 2
    "r3": (1287 + PAD, 1761 - PAD),  # CALVIN ep 1
    "r4": (1892 + PAD, 2367 - PAD),  # CALVIN ep 2
}

saved = []
for row_label, (y0, y1) in ROW_SPANS.items():
    for col_label, (x0, x1) in COL_SPANS.items():
        cell = img.crop((x0, y0, x1, y1))
        name = f"qual_{row_label}_{col_label}.png"
        cell.save(OUT / name)
        saved.append(name)
        print(f"  Saved {name}  ({x1-x0}×{y1-y0} px)")

print(f"\nAll {len(saved)} frames saved to {OUT}/")
