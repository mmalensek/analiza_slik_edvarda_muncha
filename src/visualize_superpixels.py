#!/usr/bin/env python3
"""
Vizualizacija vmesnih korakov za ekstrakcijo "important colors":
- Originalna slika
- Superpixel segmentacija
- Saliency mapa
- Dominantne barve
- Važne barve
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_hex

# Import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "analize"))
from analize.color_analysis import (
    extract_colours,
    extract_important_colors,
    extract_superpixels_with_saliency,
    colorize_segments_by_mean_color,
    saliency_to_image,
)

# Theme
FIGURE_BG = "#1c1b19"
TEXT_COLOR = "#cdccca"
ACCENT = "#4f98a3"


def rgb_to_hsv(r, g, b):
    """Convert RGB to HSV color space."""
    mx = max(r, g, b)
    mn = min(r, g, b)
    diff = mx - mn
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


def resolve_path(filename, is_dir=False):
    """Try to find file/dir in current dir or parent dirs."""
    candidates = [
        filename,
        os.path.join("..", filename),
        os.path.join("../..", filename),
        os.path.join("../../..", filename),
    ]
    for cand in candidates:
        if is_dir:
            if os.path.isdir(cand):
                return os.path.abspath(cand)
        else:
            if os.path.isfile(cand):
                return os.path.abspath(cand)
    raise FileNotFoundError(f"Could not find: {filename}")


def load_and_resize(path: str, max_px: int = 600) -> np.ndarray:
    """Load image and resize for analysis."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    return np.array(img)


def visualize_extraction_steps(image_path, output_path=None):
    """
    Visualize all intermediate steps of important color extraction.
    
    Shows:
    1. Original image
    2. Superpixel segmentation
    3. Saliency map
    4. Dominant colors
    5. Important colors
    """
    
    print(f"Loading: {os.path.basename(image_path)}")
    pixels = load_and_resize(image_path, 600)
    
    print("Extracting intermediate steps...")
    
    # Step 1: Dominant colors
    dom_colors, dom_props = extract_colours(pixels, n=6)
    
    # Step 2: Superpixel segmentation & saliency
    segments, saliency = extract_superpixels_with_saliency(pixels, n_superpixels=300)
    colored_segments = colorize_segments_by_mean_color(pixels, segments)
    saliency_img = saliency_to_image(saliency)
    
    # Step 3: Important colors
    important = extract_important_colors(pixels, n_colors=8, n_superpixels=300)
    
    n_superpixels = int(segments.max()) + 1
    print(f"Found {n_superpixels} superpixels, {len(important)} important colors")
    
    # Create figure
    fig = plt.figure(figsize=(18, 10), facecolor=FIGURE_BG)
    fig.suptitle("Important Colors Extraction Process", color=TEXT_COLOR, fontsize=16, fontweight="bold")
    
    # Plot 1: Original image
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(pixels / 255.0)
    ax1.set_title("1. Original Image", color=TEXT_COLOR, fontsize=11)
    ax1.axis("off")
    
    # Plot 2: Superpixel segmentation
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(colored_segments)
    ax2.set_title(f"2. Superpixels ({n_superpixels} regions)", color=TEXT_COLOR, fontsize=11)
    ax2.axis("off")
    
    # Plot 3: Saliency map (gradient magnitude)
    ax3 = plt.subplot(2, 3, 3)
    saliency_display = plt.cm.hot(saliency)
    ax3.imshow(saliency_display)
    ax3.set_title("3. Saliency Map (edges/contrast)", color=TEXT_COLOR, fontsize=11)
    ax3.axis("off")
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='hot'), ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Gradient", color=TEXT_COLOR, fontsize=8)
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    
    # Plot 4: Dominant colors bar
    ax4 = plt.subplot(2, 3, 4)
    y_pos = np.arange(len(dom_colors))
    ax4.barh(y_pos, dom_props * 100, color=dom_colors, edgecolor="#393836", linewidth=0.5)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([f"{rgb_to_name(c)}" for c in dom_colors], color=TEXT_COLOR, fontsize=9)
    ax4.set_xlabel("Proportion (%)", color=TEXT_COLOR, fontsize=9)
    ax4.set_title("4. Dominant Colors (K-means)", color=TEXT_COLOR, fontsize=11)
    ax4.invert_yaxis()
    ax4.set_facecolor(FIGURE_BG)
    ax4.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax4.spines.values():
        spine.set_edgecolor("#393836")
    
    # Plot 5: Important colors with scores
    ax5 = plt.subplot(2, 3, 5)
    if important:
        imp_colors = [c["rgb"] for c in important]
        imp_scores = [c["importance"] for c in important]
        y_pos_imp = np.arange(len(important))
        
        ax5.barh(y_pos_imp, imp_scores, color=imp_colors, edgecolor="#393836", linewidth=0.5)
        
        labels_imp = [
            f"{i+1}. {c['importance']:.3f}"
            for i, c in enumerate(important)
        ]
        ax5.set_yticks(y_pos_imp)
        ax5.set_yticklabels(labels_imp, color=TEXT_COLOR, fontsize=8)
        ax5.set_xlabel("Importance Score", color=TEXT_COLOR, fontsize=9)
        ax5.set_xlim(0, 1.0)
        ax5.invert_yaxis()
    
    ax5.set_title("5. Important Colors (ranked by saliency+rarity+saturation)", color=TEXT_COLOR, fontsize=11)
    ax5.set_facecolor(FIGURE_BG)
    ax5.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax5.spines.values():
        spine.set_edgecolor("#393836")
    
    # Plot 6: Detailed metrics for top important color
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")
    
    if important:
        top_color = important[0]
        info_text = f"""TOP IMPORTANT COLOR

RGB: ({top_color['rgb'][0]:.2f}, {top_color['rgb'][1]:.2f}, {top_color['rgb'][2]:.2f})

Importance Score: {top_color['importance']:.3f}

Components:
  • Saliency:   {top_color['saliency_mean']:.3f} (35% weight)
  • Rarity:     {top_color['rarity']:.3f} (25% weight)
  • Saturation: {top_color['saturation']:.3f} (20% weight)
  • Area:       {top_color['area_frac']:.3f} (20% weight)

This color's importance comes from:
- High gradient mag. at region edges
- Distinctive color (far from dominant)
- Vivid, saturated appearance
"""
    else:
        info_text = "No important colors found"
    
    ax6.text(0.1, 0.9, info_text, transform=ax6.transAxes, 
             color=TEXT_COLOR, fontsize=9, verticalalignment="top",
             family="monospace", bbox=dict(boxstyle="round", 
             facecolor="#2d2c2a", edgecolor="#393836", linewidth=1))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, facecolor=FIGURE_BG, dpi=150, bbox_inches="tight")
        print(f"Saved to: {output_path}")
    
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize intermediate steps of important color extraction."
    )
    parser.add_argument("image", nargs="?", default=None, 
                        help="Path to single image file.")
    parser.add_argument("--output", "-o", help="Save figure to file.")
    parser.add_argument("--sample", action="store_true",
                        help="Visualize a sample image from munch_paintings/ folder.")
    
    args = parser.parse_args()
    
    if args.sample:
        # Find first available painting
        paintings_dir = resolve_path("munch_paintings", is_dir=True)
        paintings = sorted([f for f in os.listdir(paintings_dir) 
                           if f.lower().endswith((".jpg", ".jpeg", ".png", ".jfif"))])
        if not paintings:
            print("No paintings found!")
            return
        image_path = os.path.join(paintings_dir, paintings[0])
        print(f"Using sample: {paintings[0]}")
    elif args.image:
        image_path = args.image
    else:
        parser.print_help()
        return
    
    visualize_extraction_steps(image_path, output_path=args.output)


if __name__ == "__main__":
    main()
