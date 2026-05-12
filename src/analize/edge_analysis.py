#!/usr/bin/env python3
"""Modul za detekcijo robov in preproste metrike teksture.
Vrne slovar osnovnih teksturnih metrik, ki jih lahko agregiramo po letu.
"""

import numpy as np
from scipy import ndimage as ndi


def compute_texture_metrics(pixels):
    # Convert to grayscale (float 0..1)
    gray = np.dot(pixels[..., :3], [0.299, 0.587, 0.114])
    gray = gray / 255.0

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

    return {
        "mean_gradient": mean_grad,
        "std_gradient": std_grad,
        "edge_density": edge_density,
        "laplacian_variance": lap_var,
    }


__all__ = ["compute_texture_metrics"]
