#!/usr/bin/env python3
"""
Časovna analiza barvnih trendov v slikah Edvarda Muncha. 

Iz vsake slike izvlečemo dominantne barve in jih združimo po letih nastanka, 
da vidimo, kako se je spreminjala svetloba, toplina, nasičenost in kompleksnost barv skozi čas.

Usage:
    python casovna_analiza.py --start 1 --end 500
"""

import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict
from scipy.stats import entropy
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
N_COLORS = 6
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".jfif")

# ---------------- IO ----------------
def collect_paths_from_range(start, end, folder):
    indexed = {}
    for name in os.listdir(folder):
        base, ext = os.path.splitext(name)
        if ext.lower() not in IMAGE_EXTS:
            continue
        if not base.isdigit():
            continue
        num = int(base)
        if start <= num <= end:
            indexed[num] = os.path.join(folder, name)
    return [indexed[k] for k in sorted(indexed)]

def load_metadata(csv_path):
    df = pd.read_csv(csv_path)
    # Normalise headers because datasets often use "number" instead of "id".
    df.columns = [str(col).strip().lower() for col in df.columns]

    if "id" not in df.columns:
        if "number" in df.columns:
            df = df.rename(columns={"number": "id"})
        else:
            raise ValueError(
                f"CSV must contain an 'id' or 'number' column. Found: {list(df.columns)}"
            )

    if "year" not in df.columns:
        raise ValueError("CSV must contain a 'year' column.")

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(int)

    # Keep the first 4-digit year (e.g., 1881 from "1881-82" or "1881–82").
    df["year"] = (
        df["year"]
        .astype(str)
        .str.extract(r"(\d{4})", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    return df.set_index("id")

def load_image(path, max_px=300):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    return np.array(img)

# ---------------- EXTERNAL ANALYSIS MODULES ----------------
from analize.color_analysis import extract_colours, brightness, warmth, saturation
from analize.edge_analysis import compute_texture_metrics

# ---------------- ANALYSIS ----------------
def analyse_painting(path, metadata, mode="both"):
    """Analiziraj eno sliko; če je potrebna tekstura, izračuna tudi texture metrike.

    mode: 'color', 'edge', or 'both'
    """
    img_id = int(os.path.splitext(os.path.basename(path))[0])

    if img_id not in metadata.index:
        return None

    year = metadata.loc[img_id]["year"]

    try:
        pixels = load_image(path)
    except Exception:
        return None

    cols = None
    weights = None
    texture = None

    if mode in ("color", "both"):
        try:
            cols, weights = extract_colours(pixels)
        except Exception:
            cols, weights = None, None

    if mode in ("edge", "both"):
        try:
            texture = compute_texture_metrics(pixels)
        except Exception:
            texture = None

    return {
        "year": year,
        "colours": cols,
        "weights": weights,
        "texture": texture,
    }

def aggregate_by_year(analyses):
    yearly = defaultdict(list)

    for a in analyses:
        if a:
            yearly[a["year"]].append(a)

    result = {}

    for year, items in yearly.items():
        cols = []
        weights = []
        textures = []

        for it in items:
            # colors may be None when running in `edge` mode
            if it.get("colours") is not None and it.get("weights") is not None:
                for c, w in zip(it["colours"], it["weights"]):
                    cols.append(c)
                    weights.append(w)
            if it.get("texture") is not None:
                textures.append(it["texture"])

        result[year] = {
            "cols": np.array(cols) if cols else np.array([]),
            "weights": np.array(weights) if weights else np.array([]),
            "textures": textures,
        }

    return result

# ---------------- TREND COMPUTATION ----------------
def compute_trends(yearly_data, mode="both"):
    """Compute trends. Returns (years, color_trends, texture_trends).

    color_trends is a dict with keys: brightness, warmth, saturation, entropy (or None if not computed).
    texture_trends is a dict with texture metrics (or None if not computed).
    """
    years = sorted(yearly_data.keys())

    color_trends = None
    texture_trends = None

    if mode in ("color", "both"):
        brightness_trend = []
        warmth_trend = []
        saturation_trend = []
        entropy_trend = []
        color_trends = {
            "brightness": brightness_trend,
            "warmth": warmth_trend,
            "saturation": saturation_trend,
            "entropy": entropy_trend,
        }

    if mode in ("edge", "both"):
        texture_trends = {
            "mean_gradient": [],
            "std_gradient": [],
            "edge_density": [],
            "laplacian_variance": [],
        }

    for y in years:
        entry = yearly_data[y]

        # Color trends
        if color_trends is not None:
            cols = entry.get("cols", np.array([]))
            weights = entry.get("weights", np.array([]))

            if len(cols) == 0 or len(weights) == 0:
                color_trends["brightness"].append(np.nan)
                color_trends["warmth"].append(np.nan)
                color_trends["saturation"].append(np.nan)
                color_trends["entropy"].append(np.nan)
            else:
                b = np.average([brightness(c) for c in cols], weights=weights)
                w = np.average([warmth(c) for c in cols], weights=weights)
                s = np.average([saturation(c) for c in cols], weights=weights)
                e = entropy(weights)

                color_trends["brightness"].append(b)
                color_trends["warmth"].append(w)
                color_trends["saturation"].append(s)
                color_trends["entropy"].append(e)

        # Texture trends
        if texture_trends is not None:
            textures = entry.get("textures", [])
            if textures:
                mg = np.mean([t["mean_gradient"] for t in textures])
                sg = np.mean([t["std_gradient"] for t in textures])
                ed = np.mean([t["edge_density"] for t in textures])
                lv = np.mean([t["laplacian_variance"] for t in textures])
            else:
                mg = sg = ed = lv = np.nan

            texture_trends["mean_gradient"].append(mg)
            texture_trends["std_gradient"].append(sg)
            texture_trends["edge_density"].append(ed)
            texture_trends["laplacian_variance"].append(lv)

    return years, color_trends, texture_trends

# ---------------- VISUALIZATION ----------------
def plot_trends(years, color_trends=None, texture_trends=None, mode="both"):
    if mode in ("color", "both") and color_trends is not None:
        plt.figure(figsize=(12, 6))
        plt.plot(years, color_trends["brightness"], label="Brightness")
        plt.plot(years, color_trends["warmth"], label="Warmth (Red - Blue)")
        plt.plot(years, color_trends["saturation"], label="Saturation")
        plt.plot(years, color_trends["entropy"], label="Colour Complexity (Entropy)")

        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.title("Temporal Colour Evolution")
        plt.legend()
        plt.grid(alpha=0.2)

        plt.tight_layout()
        plt.show()

    if mode in ("edge", "both") and texture_trends is not None:
        plt.figure(figsize=(12, 6))
        plt.plot(years, texture_trends["mean_gradient"], label="Mean Gradient")
        plt.plot(years, texture_trends["std_gradient"], label="Std Gradient")
        plt.plot(years, texture_trends["edge_density"], label="Edge Density")
        plt.plot(years, texture_trends["laplacian_variance"], label="Laplacian Variance")

        plt.xlabel("Year")
        plt.ylabel("Texture Metric")
        plt.title("Temporal Texture / Edge Trends")
        plt.legend()
        plt.grid(alpha=0.2)

        plt.tight_layout()
        plt.show()

# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--folder", default="../../munch_paintings")
    parser.add_argument("--csv", default="../data/edvard_munch.csv")
    parser.add_argument("--mode", choices=["color", "edge", "both"], default="both", help="Which analysis to run")
    args = parser.parse_args()

    print("Loading metadata...")
    metadata = load_metadata(args.csv)

    print("Collecting images...")
    paths = collect_paths_from_range(args.start, args.end, args.folder)

    print(f"Found {len(paths)} images")

    analyses = []
    for p in paths:
        print(f"Analysing {os.path.basename(p)}")
        res = analyse_painting(p, metadata, mode=args.mode)
        if res:
            analyses.append(res)

    print("Aggregating by year...")
    yearly = aggregate_by_year(analyses)

    print("Computing trends...")
    years, color_trends, texture_trends = compute_trends(yearly, mode=args.mode)

    print("Plotting results...")
    plot_trends(years, color_trends, texture_trends, mode=args.mode)

    print("Done.")

if __name__ == "__main__":
    main()