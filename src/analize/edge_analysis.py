#!/usr/bin/env python3
"""Modul za detekcijo robov in preproste metrike teksture.
Vrne slovar osnovnih teksturnih metrik, ki jih lahko agregiramo po letu.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line


def _to_gray(pixels):
    """Convert RGB image to grayscale float image in range 0..1."""
    gray = np.dot(pixels[..., :3], [0.299, 0.587, 0.114])
    return gray / 255.0


def _line_support_mask(lines, shape):
    """Rasterize line segments into a boolean mask."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for (x0, y0), (x1, y1) in lines:
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.linspace(x0, x1, n).astype(int)
        ys = np.linspace(y0, y1, n).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        mask[ys[valid], xs[valid]] = True
    return mask


def _sample_line(x0, y0, x1, y1, h, w):
    """Sample integer coordinates on a segment."""
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    xi = np.clip(np.rint(xs).astype(int), 0, w - 1)
    yi = np.clip(np.rint(ys).astype(int), 0, h - 1)
    return xi, yi


def _line_quality_score(line, gray, grad, sx, sy):
    """Compute quality score for a candidate line segment.

    Score combines:
    - gradient strength on the segment
    - cross-line contrast (left vs right side)
    - orientation consistency with local gradients
    - normalized segment length
    """
    (x0, y0), (x1, y1) = line
    h, w = gray.shape

    # Segment coordinates
    xi, yi = _sample_line(x0, y0, x1, y1, h, w)

    # Segment geometry
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    length = float(np.hypot(dx, dy))
    if length < 8:
        return 0.0

    # Unit normal vector to the line (for cross-line contrast)
    nx = -dy / (length + 1e-12)
    ny = dx / (length + 1e-12)
    off = 2.0

    x_left = np.clip(np.rint(xi + off * nx).astype(int), 0, w - 1)
    y_left = np.clip(np.rint(yi + off * ny).astype(int), 0, h - 1)
    x_right = np.clip(np.rint(xi - off * nx).astype(int), 0, w - 1)
    y_right = np.clip(np.rint(yi - off * ny).astype(int), 0, h - 1)

    # 1) Gradient strength on-line
    g_on = float(np.mean(grad[yi, xi]))

    # 2) Cross-line contrast
    contrast = float(np.mean(np.abs(gray[y_left, x_left] - gray[y_right, x_right])))

    # 3) Orientation consistency: gradient should align with line normal
    gx = sx[yi, xi]
    gy = sy[yi, xi]
    gm = np.hypot(gx, gy) + 1e-12
    dot = np.abs((gx * nx + gy * ny) / gm)
    orient_consistency = float(np.mean(dot))

    # 4) Length prior
    length_norm = float(min(1.0, length / (0.35 * min(h, w))))

    # Normalize gradient and contrast into stable ranges
    g_norm = g_on / (g_on + 0.05)
    c_norm = contrast / (contrast + 0.08)

    # Weighted score
    score = 0.35 * g_norm + 0.35 * c_norm + 0.20 * orient_consistency + 0.10 * length_norm
    return float(score)


def _filter_lines(lines, gray, grad, sx, sy, min_score=0.48):
    """Filter Hough lines to keep only perceptually meaningful edges."""
    if not lines:
        return []

    scored = []
    for line in lines:
        score = _line_quality_score(line, gray, grad, sx, sy)
        if score >= min_score:
            scored.append((line, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [line for line, _ in scored]


def detect_line_segments(pixels):
    """Detect straight line segments with probabilistic Hough transform
    using multiple line-length scales.
    """
    gray = _to_gray(pixels)
    h, w = gray.shape

    # Mild smoothing reduces brush-stroke noise before Canny/Hough.
    blur = ndi.gaussian_filter(gray, sigma=1.0)

    sx = ndi.sobel(blur, axis=0, mode="reflect")
    sy = ndi.sobel(blur, axis=1, mode="reflect")
    grad = np.hypot(sx, sy)

    canny_edges = canny(blur, sigma=1.2)

    # Different line-length scales
    base = min(h, w)

    line_lengths = [
        max(base // 40, 5),    # tiny details
        max(base // 25, 8),    # very short
        max(base // 18, 12),   # short
        max(base // 12, 18),   # medium-short
        max(base // 8, 25),    # medium
        max(base // 6, 35),    # medium-long
        max(base // 4, 50),    # long
        max(base // 3, 70),    # very long
    ]

    all_candidates = []

    for line_len in line_lengths:
        candidates = probabilistic_hough_line(
            canny_edges,
            threshold=10,
            line_length=line_len,
            line_gap=max(2, line_len // 10),
        )

        all_candidates.extend(candidates)

    # Remove duplicates / near duplicates
    unique = []
    seen = set()

    for line in all_candidates:
        (x0, y0), (x1, y1) = line

        # normalize orientation so reversed lines count the same
        if (x0, y0) > (x1, y1):
            x0, y0, x1, y1 = x1, y1, x0, y0

        key = (
            round(x0 / 5),
            round(y0 / 5),
            round(x1 / 5),
            round(y1 / 5),
        )

        if key not in seen:
            seen.add(key)
            unique.append(((x0, y0), (x1, y1)))

    return _filter_lines(unique, blur, grad, sx, sy)


def compute_texture_metrics(pixels):
    # Convert to grayscale (float 0..1)
    gray = _to_gray(pixels)

    # Sobel gradients
    sx = ndi.sobel(gray, axis=0, mode="reflect")
    sy = ndi.sobel(gray, axis=1, mode="reflect")
    grad = np.hypot(sx, sy)

    mean_grad = float(np.mean(grad))
    std_grad = float(np.std(grad))

    # Edge density: fraction of pixels stronger than mean+0.5*std
    thresh = np.mean(grad) + 0.5 * np.std(grad)
    edge_density = float((grad > thresh).mean())

    # Laplacian variance (measure of fine detail)
    lap = ndi.laplace(gray, mode="reflect")
    lap_var = float(np.var(lap))

    # Orientation-based structure metrics
    edge_mask = grad > (mean_grad + 0.5 * std_grad)
    edge_count = int(edge_mask.sum())

    orientation_entropy = 0.0
    dominant_line_orientation_deg = 0.0
    dominant_orientation_strength = 0.0

    if edge_count > 0:
        # Orientation in [0, pi): direction of edge tangent is periodic by 180 deg.
        theta = np.mod(np.arctan2(sy[edge_mask], sx[edge_mask]), np.pi)
        weights = grad[edge_mask]
        bins = 12
        hist, edges = np.histogram(theta, bins=bins, range=(0, np.pi), weights=weights)
        hist_sum = hist.sum()

        if hist_sum > 0:
            p = hist / hist_sum
            eps = 1e-12
            orientation_entropy = float(-(p * np.log(p + eps)).sum() / np.log(bins))
            idx = int(np.argmax(hist))
            center = 0.5 * (edges[idx] + edges[idx + 1])
            dominant_line_orientation_deg = float(np.degrees(center))
            dominant_orientation_strength = float(hist[idx] / hist_sum)

    # Straight-line detection (Hough)
    lines = detect_line_segments(pixels)
    num_lines = int(len(lines))

    if num_lines > 0:
        lengths = [np.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in lines]
        mean_line_length = float(np.mean(lengths))
        line_pixels = _line_support_mask(lines, gray.shape)
        if edge_count > 0:
            line_support_ratio = float((line_pixels & edge_mask).sum() / edge_count)
        else:
            line_support_ratio = 0.0
    else:
        mean_line_length = 0.0
        line_support_ratio = 0.0

    # Curve proxy: edges not explained well by straight lines
    curve_edge_ratio = float(np.clip(1.0 - line_support_ratio, 0.0, 1.0))
    curvature_index = float(curve_edge_ratio * orientation_entropy)

    return {
        "mean_gradient": mean_grad,
        "std_gradient": std_grad,
        "edge_density": edge_density,
        "laplacian_variance": lap_var,
        "num_lines": num_lines,
        "mean_line_length": mean_line_length,
        "line_support_ratio": line_support_ratio,
        "curve_edge_ratio": curve_edge_ratio,
        "orientation_entropy": orientation_entropy,
        "dominant_line_orientation_deg": dominant_line_orientation_deg,
        "dominant_orientation_strength": dominant_orientation_strength,
        "curvature_index": curvature_index,
        "hough_lines": lines,
    }


__all__ = ["compute_texture_metrics", "detect_line_segments"]
