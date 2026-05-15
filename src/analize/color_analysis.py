#!/usr/bin/env python3
"""Barvna analiza: ekstrakcija dominantnih barv in osnovni barvni metrični.
Ta modul omogoča ponovno uporabo v drugih skriptah.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import entropy
from scipy import ndimage as ndi
from skimage import segmentation, color as skcolor


N_COLORS = 6


def extract_colours(pixels, n=N_COLORS):
    flat = pixels.reshape(-1, 3).astype(float)
    km = KMeans(n_clusters=n, random_state=42, n_init=10)
    km.fit(flat)

    counts = np.bincount(km.labels_)
    order = np.argsort(-counts)

    colours = km.cluster_centers_[order] / 255.0
    weights = counts[order] / counts.sum()

    return colours, weights


def brightness(rgb):
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def warmth(rgb):
    r, g, b = rgb
    return r - b


def saturation(rgb):
    r, g, b = rgb
    mx = max(rgb)
    mn = min(rgb)
    if mx == 0:
        return 0
    return (mx - mn) / mx


# ==================== COLOR SPACE CONVERSIONS ====================
def rgb_to_lab(rgb):
    """Convert RGB (0-1) to Lab color space."""
    # Normalize to 0-1 if needed
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    # Use scikit-image for accurate conversion
    try:
        return skcolor.rgb2lab(np.array([rgb]))[0]
    except Exception:
        return rgb


def lab_distance(lab1, lab2):
    """Euclidean distance in Lab space (perceptual)."""
    return np.sqrt(np.sum((lab1 - lab2) ** 2))


def cluster_similar_colors(colors, frequencies=None, lab_threshold=15.0):
    """
    Cluster similar colors in LAB space to deduplicate similar hues.
    
    Args:
        colors: array of RGB colors (0-1), shape (N, 3)
        frequencies: optional array of frequencies/importances for each color
        lab_threshold: Lab distance threshold for grouping similar colors
    
    Returns:
        merged_colors: deduplicated RGB colors
        merged_freqs: summed frequencies for merged colors
    """
    if len(colors) == 0:
        return np.array([]), np.array([])
    
    colors = np.array(colors)
    if frequencies is None:
        frequencies = np.ones(len(colors))
    frequencies = np.array(frequencies)
    
    # Convert to Lab
    colors_lab = np.array([rgb_to_lab(c) for c in colors])
    
    # Greedy clustering: go through colors, merge similar ones
    merged = []
    merged_freqs = []
    used = set()
    
    for i in range(len(colors)):
        if i in used:
            continue
        
        # Start a new cluster with this color
        cluster_rgb = [colors[i]]
        cluster_freq = [frequencies[i]]
        used.add(i)
        
        # Find all similar colors
        for j in range(i + 1, len(colors)):
            if j not in used:
                dist = lab_distance(colors_lab[i], colors_lab[j])
                if dist < lab_threshold:
                    cluster_rgb.append(colors[j])
                    cluster_freq.append(frequencies[j])
                    used.add(j)
        
        # Merge cluster: weighted average color
        merged_color = np.average(cluster_rgb, axis=0, weights=cluster_freq)
        merged_freq = sum(cluster_freq)
        
        merged.append(merged_color)
        merged_freqs.append(merged_freq)
    
    return np.array(merged), np.array(merged_freqs)


# ==================== SALIENCY COMPUTATION ====================
def compute_saliency(pixels):
    """
    Compute saliency map using gradient-based method.
    Returns normalized saliency map (0-1).
    """
    gray = np.dot(pixels[..., :3], [0.299, 0.587, 0.114]) / 255.0

    # Gradient magnitude (Sobel)
    sx = ndi.sobel(gray, axis=0, mode="reflect")
    sy = ndi.sobel(gray, axis=1, mode="reflect")
    saliency = np.hypot(sx, sy)

    # Normalize to 0-1
    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    return saliency


# ==================== IMPORTANT COLORS ====================
def extract_important_colors(pixels, n_colors=10, n_superpixels=300):
    """
    Extract important colors from image using saliency, contrast, and rarity.

    Returns: list of dicts with keys:
        - 'rgb': RGB color (0-1)
        - 'area_frac': fraction of image this color occupies
        - 'saliency_mean': avg saliency in region
        - 'saturation': color saturation
        - 'rarity': how rare/different from dominant colors
        - 'importance': overall importance score
    """
    h, w = pixels.shape[:2]

    # Compute saliency
    saliency = compute_saliency(pixels)

    # Superpixel segmentation (SLIC)
    try:
        segments = segmentation.slic(
            pixels, n_segments=n_superpixels, sigma=1, start_label=0, compactness=10
        )
    except Exception:
        # Fallback: simple grid segmentation
        segments = np.zeros((h, w), dtype=int)
        step = max(h // int(np.sqrt(n_superpixels)), 1)
        idx = 0
        for i in range(0, h, step):
            for j in range(0, w, step):
                segments[i : i + step, j : j + step] = idx
                idx += 1

    # Get dominant colors for reference
    dominant_colors, _ = extract_colours(pixels, n=N_COLORS)
    dominant_lab = np.array([rgb_to_lab(c) for c in dominant_colors])

    # Analyze each superpixel
    superpixel_data = []
    for seg_id in np.unique(segments):
        mask = segments == seg_id
        region_pixels = pixels[mask]

        if len(region_pixels) == 0:
            continue

        # Compute region statistics
        area_frac = mask.sum() / (h * w)
        saliency_mean = saliency[mask].mean()
        color_mean_rgb = region_pixels.mean(axis=0) / 255.0  # 0-1
        color_mean_lab = rgb_to_lab(color_mean_rgb)

        # Distance to nearest dominant color in Lab
        min_dist_to_dominant = min(
            [lab_distance(color_mean_lab, d) for d in dominant_lab]
        )

        # Saturation
        sat = saturation(color_mean_rgb)

        # Rarity: inverse of closeness to dominant colors
        rarity = min_dist_to_dominant / (1.0 + min_dist_to_dominant)  # 0-1

        # Importance score (weighted combination)
        importance = (
            0.35 * saliency_mean  # how visually prominent
            + 0.25 * rarity  # how different from dominant
            + 0.20 * sat  # how vivid/saturated
            + 0.20 * min(area_frac, 0.5)  # area (diminishing returns after 50%)
        )

        superpixel_data.append(
            {
                "rgb": color_mean_rgb,
                "lab": color_mean_lab,
                "area_frac": area_frac,
                "saliency_mean": saliency_mean,
                "saturation": sat,
                "rarity": rarity,
                "importance": importance,
            }
        )

    # Sort by importance descending
    superpixel_data.sort(key=lambda x: x["importance"], reverse=True)

    # Deduplication: merge very similar colors (Lab distance < 10)
    final_colors = []
    for data in superpixel_data:
        # Check if similar color already exists
        is_duplicate = False
        for existing in final_colors:
            if lab_distance(data["lab"], existing["lab"]) < 10:
                # Keep the one with higher importance
                if data["importance"] > existing["importance"]:
                    final_colors.remove(existing)
                    final_colors.append(data)
                is_duplicate = True
                break

        if not is_duplicate:
            final_colors.append(data)

    # Return top N
    return final_colors[:n_colors]


# ==================== VISUALIZATION HELPERS ====================
def extract_superpixels_with_saliency(pixels, n_superpixels=300):
    """
    Extract superpixel segmentation and saliency map.
    Returns: (segments, saliency) arrays suitable for visualization.
    """
    h, w = pixels.shape[:2]
    
    # Compute saliency
    saliency = compute_saliency(pixels)
    
    # Superpixel segmentation
    try:
        segments = segmentation.slic(
            pixels, n_segments=n_superpixels, sigma=1, start_label=0, compactness=10
        )
    except Exception:
        segments = np.zeros((h, w), dtype=int)
        step = max(h // int(np.sqrt(n_superpixels)), 1)
        idx = 0
        for i in range(0, h, step):
            for j in range(0, w, step):
                segments[i : i + step, j : j + step] = idx
                idx += 1
    
    return segments, saliency


def colorize_segments(segments):
    """
    Create a colored visualization of superpixel segments.
    Each segment gets a random color. Returns RGB image (0-1).
    """
    # Create pseudocolor image (rainbow map for segment IDs)
    max_seg = segments.max() + 1
    cmap = plt.cm.get_cmap("hsv", max_seg)
    colored = cmap(segments / max_seg)[:, :, :3]  # Drop alpha channel
    return colored


def colorize_segments_by_mean_color(pixels, segments):
    """
    Color superpixel segments by their mean RGB color from the image.
    Returns RGB image (0-1) where each superpixel is colored with its mean color.
    """
    h, w = pixels.shape[:2]
    colored = np.zeros((h, w, 3), dtype=np.float32)
    
    # For each superpixel, compute and assign mean color
    for seg_id in np.unique(segments):
        mask = segments == seg_id
        region_pixels = pixels[mask]
        
        if len(region_pixels) > 0:
            mean_color = region_pixels.mean(axis=0) / 255.0  # Normalize to 0-1
            colored[mask] = mean_color
    
    return colored


def saliency_to_image(saliency):
    """Convert saliency map (0-1) to grayscale image for display."""
    return np.stack([saliency] * 3, axis=-1)


__all__ = [
    "extract_colours",
    "brightness",
    "warmth",
    "saturation",
    "rgb_to_lab",
    "lab_distance",
    "cluster_similar_colors",
    "compute_saliency",
    "extract_important_colors",
    "extract_superpixels_with_saliency",
    "colorize_segments",
    "colorize_segments_by_mean_color",
    "saliency_to_image",
    "N_COLORS",
]
