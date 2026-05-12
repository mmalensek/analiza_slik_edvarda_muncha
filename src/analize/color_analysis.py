#!/usr/bin/env python3
"""Barvna analiza: ekstrakcija dominantnih barv in osnovni barvni metrični.
Ta modul omogoča ponovno uporabo v drugih skriptah.
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import entropy

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


__all__ = ["extract_colours", "brightness", "warmth", "saturation", "N_COLORS"]
