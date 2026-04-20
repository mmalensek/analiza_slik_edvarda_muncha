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

# ---------------- COLOR EXTRACTION ----------------
def extract_colours(pixels, n=N_COLORS):
    flat = pixels.reshape(-1, 3).astype(float)
    km = KMeans(n_clusters=n, random_state=42, n_init=10)
    km.fit(flat)

    counts = np.bincount(km.labels_)
    order = np.argsort(-counts)

    colours = km.cluster_centers_[order] / 255.0
    weights = counts[order] / counts.sum()

    return colours, weights

# ---------------- METRICS ----------------
def brightness(rgb):
    r, g, b = rgb
    return 0.299*r + 0.587*g + 0.114*b

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

# ---------------- ANALYSIS ----------------
def analyse_painting(path, metadata):
    img_id = int(os.path.splitext(os.path.basename(path))[0])

    if img_id not in metadata.index:
        return None

    year = metadata.loc[img_id]["year"]

    try:
        pixels = load_image(path)
        cols, weights = extract_colours(pixels)
    except Exception:
        return None

    return {
        "year": year,
        "colours": cols,
        "weights": weights
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

        for it in items:
            for c, w in zip(it["colours"], it["weights"]):
                cols.append(c)
                weights.append(w)

        result[year] = (np.array(cols), np.array(weights))

    return result

# ---------------- TREND COMPUTATION ----------------
def compute_trends(yearly_data):
    years = sorted(yearly_data.keys())

    brightness_trend = []
    warmth_trend = []
    saturation_trend = []
    entropy_trend = []

    for y in years:
        cols, weights = yearly_data[y]

        b = np.average([brightness(c) for c in cols], weights=weights)
        w = np.average([warmth(c) for c in cols], weights=weights)
        s = np.average([saturation(c) for c in cols], weights=weights)
        e = entropy(weights)

        brightness_trend.append(b)
        warmth_trend.append(w)
        saturation_trend.append(s)
        entropy_trend.append(e)

    return years, brightness_trend, warmth_trend, saturation_trend, entropy_trend

# ---------------- VISUALIZATION ----------------
def plot_trends(years, brightness, warmth, saturation, entropy_vals):
    plt.figure(figsize=(12, 6))

    plt.plot(years, brightness, label="Brightness")
    plt.plot(years, warmth, label="Warmth (Red - Blue)")
    plt.plot(years, saturation, label="Saturation")
    plt.plot(years, entropy_vals, label="Colour Complexity (Entropy)")

    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.title("Temporal Colour Evolution")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()

# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--folder", default="../munch_paintings")
    parser.add_argument("--csv", default="edvard_munch.csv")
    args = parser.parse_args()

    print("Loading metadata...")
    metadata = load_metadata(args.csv)

    print("Collecting images...")
    paths = collect_paths_from_range(args.start, args.end, args.folder)

    print(f"Found {len(paths)} images")

    analyses = []
    for p in paths:
        print(f"Analysing {os.path.basename(p)}")
        res = analyse_painting(p, metadata)
        if res:
            analyses.append(res)

    print("Aggregating by year...")
    yearly = aggregate_by_year(analyses)

    print("Computing trends...")
    years, b, w, s, e = compute_trends(yearly)

    print("Plotting results...")
    plot_trends(years, b, w, s, e)

    print("Done.")

if __name__ == "__main__":
    main()