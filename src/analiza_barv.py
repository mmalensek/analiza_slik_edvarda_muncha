#!/usr/bin/env python3
"""
Analiz barv slik Edvarda Muncha z uporabo k-means algoritma za pridobivanje dominantnih barv.

Usage: analiza_barv.py [slika1.jpg] [slika2.jpg] ...
"""

import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.colors import to_hex
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# configuration
N_COLORS = 8          # dominant colours to extract per painting
FIGURE_BG = "#1c1b19" # dark warm surface 
TEXT_COLOR = "#cdccca"
ACCENT = "#4f98a3"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".jfif", ".webp", ".bmp", ".tiff")


def collect_paths_from_range(start: int, end: int, folder: str) -> list[str]:
    """Find existing numbered images in folder within [start, end], skipping missing numbers."""
    if start > end:
        raise ValueError("Range start must be less than or equal to range end.")

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    indexed_paths: dict[int, str] = {}
    for name in os.listdir(folder):
        base, ext = os.path.splitext(name)
        if ext.lower() not in IMAGE_EXTS:
            continue
        if not base.isdigit():
            continue
        num = int(base)
        if start <= num <= end:
            indexed_paths[num] = os.path.join(folder, name)

    return [indexed_paths[n] for n in sorted(indexed_paths)]

def load_and_resize(path: str, max_px: int = 300) -> np.ndarray:
    """Load image and resize to speed up k-means."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    return np.array(img)

def extract_dominant_colours(pixels: np.ndarray, n: int = N_COLORS):
    """Run k-means on pixel RGB values, return colours sorted by frequency."""
    flat = pixels.reshape(-1, 3).astype(float)
    km = KMeans(n_clusters=n, random_state=42, n_init=10)
    km.fit(flat)
    counts = np.bincount(km.labels_)
    order = np.argsort(-counts)
    colours = (km.cluster_centers_[order] / 255.0)
    proportions = counts[order] / counts.sum()
    return colours, proportions

def rgb_to_name(rgb_norm):
    """Very rough colour naming based on hue/saturation/value."""
    r, g, b = rgb_norm
    h, s, v = rgb_to_hsv(r, g, b)
    if v < 0.15:
        return "Black"
    if v > 0.85 and s < 0.15:
        return "White"
    if s < 0.15:
        return f"Gray ({int(v*100)}% L)"
    names = [
        (0,   30,  "Red"),
        (30,  45,  "Orange"),
        (45,  65,  "Yellow"),
        (65,  160, "Green"),
        (160, 200, "Cyan"),
        (200, 260, "Blue"),
        (260, 290, "Purple"),
        (290, 330, "Magenta"),
        (330, 360, "Red"),
    ]
    for lo, hi, name in names:
        if lo <= h < hi:
            prefix = "Dark " if v < 0.4 else ("Light " if v > 0.7 else "")
            return prefix + name
    return "Red"

def rgb_to_hsv(r, g, b):
    mx = max(r, g, b); mn = min(r, g, b); diff = mx - mn
    v = mx
    s = diff / mx if mx != 0 else 0
    if diff == 0:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / diff)) % 360
    elif mx == g:
        h = 60 * ((b - r) / diff + 2)
    else:
        h = 60 * ((r - g) / diff + 4)
    return h, s, v

def analyse_painting(path: str):
    """Load image → extract colours → return dict of results."""
    print(f"  Analysing: {os.path.basename(path)} …")
    pixels = load_and_resize(path)
    colours, proportions = extract_dominant_colours(pixels)
    names = [rgb_to_name(c) for c in colours]
    hex_codes = [to_hex(c) for c in colours]
    return {
        "title": os.path.splitext(os.path.basename(path))[0].replace("_", " ").title(),
        "path": path,
        "colours": colours,
        "proportions": proportions,
        "names": names,
        "hex_codes": hex_codes,
    }

# Visualization

def animate_analyses(analyses):
    """Build an animated matplotlib figure cycling through paintings."""
    n_paintings = len(analyses)
    fig = plt.figure(figsize=(14, 7), facecolor=FIGURE_BG)
    fig.suptitle("Edvard Munch: Barvna analiza", color=TEXT_COLOR,
                 fontsize=16, fontweight="bold", y=0.97)

    # axes layout
    ax_img  = fig.add_axes([0.03, 0.08, 0.32, 0.78])   # painting preview
    ax_bar  = fig.add_axes([0.40, 0.08, 0.28, 0.78])   # horizontal bar chart
    ax_pie  = fig.add_axes([0.72, 0.12, 0.26, 0.70])   # donut chart

    for ax in [ax_img, ax_bar, ax_pie]:
        ax.set_facecolor(FIGURE_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor("#393836")

    # prepare frame data
    frame_data = []
    for a in analyses:
        for repeat in range(60):   # ~3 s at 20 fps
            frame_data.append((a, repeat))

    def draw_frame(idx):
        data, tick = frame_data[idx]
        # progress goes from 0 to 1 over the first 20 ticks
        # then stays at 1 for the remaining ticks
        progress = min(tick / 20, 1.0)

        # clear axes for redraw
        ax_img.cla(); ax_bar.cla(); ax_pie.cla()
        for ax in [ax_img, ax_bar, ax_pie]:
            ax.set_facecolor(FIGURE_BG)

        # painting image (fade in)
        try:
            img_arr = load_and_resize(data["path"], 300)
            ax_img.imshow(img_arr, alpha=min(progress * 2, 1))
        except Exception:
            ax_img.set_facecolor("#2d2c2a")
        ax_img.axis("off")
        ax_img.set_title(data["title"], color=TEXT_COLOR, fontsize=11, pad=6)

        # bar chart (animated widths)
        n = len(data["colours"])
        y_pos = np.arange(n)
        widths = data["proportions"] * 100 * progress   # animate bar growth

        bars = ax_bar.barh(y_pos, widths, color=data["colours"].tolist(),
                           edgecolor="#393836", linewidth=0.5)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(
            [f"{data['names'][i]}  {data['hex_codes'][i]}"
             for i in range(n)],
            color=TEXT_COLOR, fontsize=8)
        ax_bar.set_xlim(0, 55)
        ax_bar.set_xlabel("Proportion (%)", color=TEXT_COLOR, fontsize=8)
        ax_bar.tick_params(colors=TEXT_COLOR, labelsize=7)
        ax_bar.set_title("Dominant Colours", color=TEXT_COLOR, fontsize=10, pad=6)
        ax_bar.invert_yaxis()
        for spine in ax_bar.spines.values():
            spine.set_edgecolor("#393836")

        # add percentage labels
        for bar, pct in zip(bars, data["proportions"] * 100 * progress):
            if pct > 1.5:
                ax_bar.text(pct + 0.3, bar.get_y() + bar.get_height()/2,
                            f"{pct:.1f}%", va="center", color=TEXT_COLOR,
                            fontsize=7)

        # donut chart (animated proportions)
        animated_props = data["proportions"] * progress
        if animated_props.sum() < 0.001:
            animated_props = np.ones(n) / n * 0.001
        wedge_props = animated_props / animated_props.sum()

        wedges, _ = ax_pie.pie(
            wedge_props,
            colors=data["colours"].tolist(),
            startangle=90,
            wedgeprops=dict(width=0.55, edgecolor=FIGURE_BG, linewidth=1.5)
        )
        ax_pie.set_title("Colour Composition", color=TEXT_COLOR,
                         fontsize=10, pad=6)

        # progress indicator (dots below figure)
        fig.texts = [t for t in fig.texts if t.get_text().startswith("Edvard")]
        pidx = analyses.index(data)
        dots = "  ".join(
            ("O" if i == pidx else "I") for i in range(n_paintings)
        )
        fig.text(0.5, 0.01, dots, ha="center", color=ACCENT, fontsize=14)

        return []

    ani = animation.FuncAnimation(
        fig,
        draw_frame,
        frames=len(frame_data),
        interval=50,   # 20 fps
        blit=False,
        repeat=True
    )
    return fig, ani


# main
def main():
    print("\nBarvna analiza slik Edvarda Muncha")
    print("-" * 40)

    parser = argparse.ArgumentParser(
        description="Analyse dominant colours in Munch paintings."
    )
    parser.add_argument("images", nargs="*", help="Optional explicit image paths.")
    parser.add_argument("--start", type=int, default=None,
                        help="Start of numeric image range (inclusive).")
    parser.add_argument("--end", type=int, default=None,
                        help="End of numeric image range (inclusive).")
    parser.add_argument("--folder", default="../munch_paintings",
                        help="Folder containing numbered image files.")
    args = parser.parse_args()

    paths = []
    if args.start is not None and args.end is not None:
        try:
            paths = collect_paths_from_range(args.start, args.end, args.folder)
        except Exception as exc:
            print(f"Error while collecting images from range: {exc}")
            return
        if not paths:
            print(f"No images found in range {args.start}..{args.end} under {args.folder}.")
            return
        print(f"Found {len(paths)} images in range {args.start}..{args.end}.")
    elif args.images:
        paths = [p for p in args.images if p.lower().endswith(IMAGE_EXTS)]
        if not paths:
            print("No valid image paths found.")
            return
    else:
        print("No images supplied. Use --start/--end for range mode or pass explicit image paths.")
        return

    analyses = [analyse_painting(p) for p in paths]

    print("\nLaunching animated visualisation:")
    print("(Close the window to exit, or press Q)")
    fig, ani = animate_analyses(analyses)
    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()
