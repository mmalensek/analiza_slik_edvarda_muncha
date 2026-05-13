#!/usr/bin/env python3
"""
Analiza ene slike (ali range slik) Edvarda Muncha:
- dominantne barve (K-means)
- pomembne barve (saliency + rarity + saturation)
- robovi / tekstura (Sobel, Laplacian)

Usage: analiza_ene_slike.py [slika1.jpg] [slika2.jpg] ...
       analiza_ene_slike.py --start 1 --end 20
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
from scipy import ndimage as ndi
import warnings
warnings.filterwarnings("ignore")

# Import analysis modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "analize"))
from analize.color_analysis import extract_important_colors, compute_saliency
from analize.edge_analysis import compute_texture_metrics

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

def compute_edge_map(pixels):
    """Compute edge magnitude map using Sobel."""
    gray = np.dot(pixels[..., :3], [0.299, 0.587, 0.114]) / 255.0
    sx = ndi.sobel(gray, axis=0, mode="reflect")
    sy = ndi.sobel(gray, axis=1, mode="reflect")
    edges = np.hypot(sx, sy)
    if edges.max() > 0:
        edges = edges / edges.max()
    return edges

def analyse_painting(path: str):
    """Load image → extract colours, important colours, texture → return dict of results."""
    print(f"  Analysing: {os.path.basename(path)} …")
    pixels = load_and_resize(path)
    
    # Dominant colours
    colours, proportions = extract_dominant_colours(pixels)
    names = [rgb_to_name(c) for c in colours]
    hex_codes = [to_hex(c) for c in colours]
    
    # Important colours (saliency + contrast + rarity)
    try:
        important_colors = extract_important_colors(pixels, n_colors=8, n_superpixels=300)
    except Exception as e:
        print(f"    Warning: Could not extract important colors: {e}")
        important_colors = []
    
    # Texture metrics
    try:
        texture = compute_texture_metrics(pixels)
        saliency = compute_saliency(pixels)
        edges = compute_edge_map(pixels)
    except Exception as e:
        print(f"    Warning: Could not compute texture: {e}")
        texture = {}
        saliency = np.zeros_like(pixels[:, :, 0])
        edges = np.zeros_like(pixels[:, :, 0])
    
    return {
        "title": os.path.splitext(os.path.basename(path))[0].replace("_", " ").title(),
        "path": path,
        # Dominant colours
        "colours": colours,
        "proportions": proportions,
        "names": names,
        "hex_codes": hex_codes,
        # Important colours
        "important_colors": important_colors,
        # Texture
        "texture": texture,
        "saliency": saliency,
        "edges": edges,
        "hough_lines": texture.get("hough_lines", []) if texture else [],
    }

# Visualization

def animate_analyses(analyses):
    """Build an animated matplotlib figure cycling through paintings."""
    n_paintings = len(analyses)
    fig = plt.figure(figsize=(16, 10), facecolor=FIGURE_BG)
    fig.suptitle("Edvard Munch: Analiza barv in teksture", color=TEXT_COLOR,
                 fontsize=16, fontweight="bold", y=0.98)

    # axes layout - 2x3 grid with better spacing
    ax_img       = fig.add_axes([0.02, 0.50, 0.27, 0.40])   # painting preview (top-left)
    ax_bar       = fig.add_axes([0.34, 0.50, 0.27, 0.40])   # dominant colours (top-middle)
    ax_important = fig.add_axes([0.66, 0.50, 0.27, 0.40])   # important colours (top-right)
    
    ax_saliency  = fig.add_axes([0.02, 0.04, 0.27, 0.40])   # saliency map (bottom-left)
    ax_edges     = fig.add_axes([0.34, 0.04, 0.27, 0.40])   # edges map (bottom-middle)
    ax_texture   = fig.add_axes([0.66, 0.04, 0.27, 0.40])   # texture metrics (bottom-right)

    for ax in [ax_img, ax_bar, ax_important, ax_saliency, ax_edges, ax_texture]:
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
        progress = min(tick / 20, 1.0)

        # clear axes for redraw
        for ax in [ax_img, ax_bar, ax_important, ax_saliency, ax_edges, ax_texture]:
            ax.cla()
            ax.set_facecolor(FIGURE_BG)

        # ============ TOP ROW ============

        # painting image (fade in)
        try:
            img_arr = load_and_resize(data["path"], 300)
            ax_img.imshow(img_arr, alpha=min(progress * 2, 1))
        except Exception:
            ax_img.set_facecolor("#2d2c2a")
        ax_img.axis("off")
        ax_img.set_title(data["title"], color=TEXT_COLOR, fontsize=11, pad=6)

        # dominant colours bar chart (animated widths)
        n = len(data["colours"])
        y_pos = np.arange(n)
        widths = data["proportions"] * 100 * progress   # animate bar growth

        bars = ax_bar.barh(y_pos, widths, color=data["colours"].tolist(),
                           edgecolor="#393836", linewidth=0.5)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(
            [f"{data['names'][i]}"
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

        # Important colours (saliency + contrast + rarity based)
        important_colors = data.get("important_colors", [])
        if important_colors:
            n_imp = len(important_colors)
            y_pos_imp = np.arange(n_imp)
            
            # Get RGB colors and importance scores
            imp_rgb_colors = [c["rgb"] for c in important_colors]
            imp_scores = np.array([c["importance"] for c in important_colors]) * progress
            
            bars_imp = ax_important.barh(y_pos_imp, imp_scores, 
                                         color=imp_rgb_colors,
                                         edgecolor="#393836", linewidth=0.5)
            
            # Labels
            labels_imp = [f"{i+1}. {important_colors[i]['importance']:.3f}" for i in range(n_imp)]
            ax_important.set_yticks(y_pos_imp)
            ax_important.set_yticklabels(labels_imp, color=TEXT_COLOR, fontsize=7)
            ax_important.set_xlim(0, 1.0)
            ax_important.set_xlabel("Importance", color=TEXT_COLOR, fontsize=8)
            ax_important.tick_params(colors=TEXT_COLOR, labelsize=7)
            ax_important.set_title("Important Colours", color=TEXT_COLOR, fontsize=10, pad=6)
            ax_important.invert_yaxis()
            for spine in ax_important.spines.values():
                spine.set_edgecolor("#393836")

        # ============ BOTTOM ROW ============

        # Saliency map (gradient magnitude heatmap)
        if data.get("saliency") is not None and data["saliency"].size > 0:
            saliency_display = plt.cm.hot(data["saliency"])
            ax_saliency.imshow(saliency_display, alpha=progress)
        ax_saliency.axis("off")
        ax_saliency.set_title("Saliency (edges/contrast)", color=TEXT_COLOR, fontsize=10, pad=6)

        # Edge magnitude map
        if data.get("edges") is not None and data["edges"].size > 0:
            edges_display = plt.cm.gray(data["edges"])
            ax_edges.imshow(edges_display, alpha=progress)
            # Overlay detected straight lines (Hough)
            for (x0, y0), (x1, y1) in data.get("hough_lines", []):
                ax_edges.plot([x0, x1], [y0, y1], color="#e53935", linewidth=1.0, alpha=0.8 * progress)
        ax_edges.axis("off")
        ax_edges.set_title("Edge Magnitude + Straight Lines", color=TEXT_COLOR, fontsize=10, pad=6)

        # Texture metrics bar chart
        texture = data.get("texture", {})
        if texture:
            metrics = ["Mean G", "Std G", "Edge D", "Lapl V", "Line Sup", "Curve R", "Ori Ent"]
            values = [
                texture.get("mean_gradient", 0),
                texture.get("std_gradient", 0),
                texture.get("edge_density", 0),
                texture.get("laplacian_variance", 0),
                texture.get("line_support_ratio", 0),
                texture.get("curve_edge_ratio", 0),
                texture.get("orientation_entropy", 0),
            ]
            # Scale to comparable visual ranges for bar chart readability
            scales = np.array([2.5, 5.0, 1.0, 80.0, 1.0, 1.0, 1.0])
            values = (np.array(values) * scales) * progress
            
            colors_tex = ["#e85d75", "#f39c12", "#3498db", "#2ecc71", "#4f98a3", "#8bc34a", "#9c27b0"]
            bars_tex = ax_texture.bar(range(len(metrics)), values, color=colors_tex,
                                      edgecolor="#393836", linewidth=0.5)
            ax_texture.set_xticks(range(len(metrics)))
            ax_texture.set_xticklabels(metrics, color=TEXT_COLOR, fontsize=7, rotation=35, ha='right')
            ax_texture.set_ylabel("Scaled Value", color=TEXT_COLOR, fontsize=8)
            ax_texture.tick_params(colors=TEXT_COLOR, labelsize=7)
            ax_texture.set_title("Texture + Line/Curve Metrics", color=TEXT_COLOR, fontsize=10, pad=6)
            
            # Add value labels
            for bar, val in zip(bars_tex, values):
                height = bar.get_height()
                if height > 0.02:
                    ax_texture.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{val:.2f}',
                                   ha='center', va='bottom', color=TEXT_COLOR, fontsize=7)

            # Small textual summary for interpretation
            straightness = texture.get("line_support_ratio", 0.0)
            curviness = texture.get("curve_edge_ratio", 0.0)
            dom_angle = texture.get("dominant_line_orientation_deg", 0.0)
            ax_texture.text(
                0.02,
                0.96,
                f"Straightness: {straightness:.2f} | Curviness: {curviness:.2f} | Dom angle: {dom_angle:.0f} deg",
                transform=ax_texture.transAxes,
                color=TEXT_COLOR,
                fontsize=7,
                va="top",
            )
        
        for spine in ax_texture.spines.values():
            spine.set_edgecolor("#393836")

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
    print("\nAnaliza slik Edvarda Muncha: barve + tekstura")
    print("-" * 50)

    parser = argparse.ArgumentParser(
        description="Analyse dominant colours, important colours, and texture in Munch paintings."
    )
    parser.add_argument("images", nargs="*", help="Optional explicit image paths.")
    parser.add_argument("--start", type=int, default=None,
                        help="Start of numeric image range (inclusive).")
    parser.add_argument("--end", type=int, default=None,
                        help="End of numeric image range (inclusive).")
    parser.add_argument("--folder", default="../../munch_paintings",
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
