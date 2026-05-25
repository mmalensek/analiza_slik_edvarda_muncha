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


# ---------------- FALLBACK METRICS ----------------
def compute_fallback_symmetry_composition(pixels):
    gray = np.array(Image.fromarray(pixels).convert("L"), dtype=np.float32) / 255.0
    h, w = gray.shape

    if w < 4 or h < 4:
        return {"symmetry_score": np.nan, "composition_score": np.nan}

    half = w // 2
    left = gray[:, :half]
    right = gray[:, w - half:]
    right_flipped = np.fliplr(right)

    min_w = min(left.shape[1], right_flipped.shape[1])
    left = left[:, :min_w]
    right_flipped = right_flipped[:, :min_w]

    sym_diff = np.mean(np.abs(left - right_flipped))
    symmetry_score = float(np.clip(1.0 - sym_diff, 0.0, 1.0))

    gy, gx = np.gradient(gray)
    energy = np.sqrt(gx ** 2 + gy ** 2)
    total_energy = energy.sum()

    if total_energy <= 1e-8:
        return {
            "symmetry_score": symmetry_score,
            "composition_score": np.nan,
        }

    ys, xs = np.indices(gray.shape)
    cx = float((xs * energy).sum() / total_energy)
    cy = float((ys * energy).sum() / total_energy)

    thirds_points = [
        (w / 3.0, h / 3.0),
        (2 * w / 3.0, h / 3.0),
        (w / 3.0, 2 * h / 3.0),
        (2 * w / 3.0, 2 * h / 3.0),
    ]

    distances = [np.hypot(cx - tx, cy - ty) for tx, ty in thirds_points]
    best_dist = min(distances)
    max_dist = np.hypot(w, h)
    thirds_alignment = 1.0 - (best_dist / max_dist)

    left_energy = energy[:, :half].sum()
    right_energy = energy[:, w - half:].sum()
    balance = 1.0 - abs(left_energy - right_energy) / max(total_energy, 1e-8)

    composition_score = float(np.clip(0.65 * thirds_alignment + 0.35 * balance, 0.0, 1.0))

    return {
        "symmetry_score": symmetry_score,
        "composition_score": composition_score,
    }


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
            if texture is None:
                texture = {}

            if texture.get("symmetry_score") is None and texture.get("symmetry") is not None:
                texture["symmetry_score"] = texture.get("symmetry")

            if texture.get("composition_score") is None and texture.get("composition_balance") is not None:
                texture["composition_score"] = texture.get("composition_balance")

            if texture.get("symmetry_score") is None or texture.get("composition_score") is None:
                texture.update(compute_fallback_symmetry_composition(pixels))
        except Exception:
            texture = compute_fallback_symmetry_composition(pixels)

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
def safe_mean(values):
    arr = np.array(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def compute_trends(yearly_data, mode="both"):
    years = sorted(yearly_data.keys())

    color_trends = None
    texture_trends = None

    if mode in ("color", "both"):
        color_trends = {
            "brightness": [],
            "warmth": [],
            "saturation": [],
            "entropy": [],
        }

    if mode in ("edge", "both"):
        texture_trends = {
            "mean_gradient": [],
            "std_gradient": [],
            "edge_density": [],
            "laplacian_variance": [],
            "num_lines": [],
            "mean_line_length": [],
            "line_support_ratio": [],
            "curve_edge_ratio": [],
            "orientation_entropy": [],
            "dominant_orientation_strength": [],
            "curvature_index": [],
            "symmetry_score": [],
            "composition_score": [],
        }

    for y in years:
        entry = yearly_data[y]

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

        if texture_trends is not None:
            textures = entry.get("textures", [])

            if textures:
                mg = safe_mean([t.get("mean_gradient", np.nan) for t in textures])
                sg = safe_mean([t.get("std_gradient", np.nan) for t in textures])
                ed = safe_mean([t.get("edge_density", np.nan) for t in textures])
                lv = safe_mean([t.get("laplacian_variance", np.nan) for t in textures])
                nl = safe_mean([t.get("num_lines", np.nan) for t in textures])
                ml = safe_mean([t.get("mean_line_length", np.nan) for t in textures])
                ls = safe_mean([t.get("line_support_ratio", np.nan) for t in textures])
                cr = safe_mean([t.get("curve_edge_ratio", np.nan) for t in textures])
                oe = safe_mean([t.get("orientation_entropy", np.nan) for t in textures])
                ds = safe_mean([t.get("dominant_orientation_strength", np.nan) for t in textures])
                ci = safe_mean([t.get("curvature_index", np.nan) for t in textures])
                ss = safe_mean([t.get("symmetry_score", t.get("symmetry", np.nan)) for t in textures])
                cs = safe_mean([t.get("composition_score", t.get("composition_balance", np.nan)) for t in textures])
            else:
                mg = sg = ed = lv = nl = ml = ls = cr = oe = ds = ci = ss = cs = np.nan

            texture_trends["mean_gradient"].append(mg)
            texture_trends["std_gradient"].append(sg)
            texture_trends["edge_density"].append(ed)
            texture_trends["laplacian_variance"].append(lv)
            texture_trends["num_lines"].append(nl)
            texture_trends["mean_line_length"].append(ml)
            texture_trends["line_support_ratio"].append(ls)
            texture_trends["curve_edge_ratio"].append(cr)
            texture_trends["orientation_entropy"].append(oe)
            texture_trends["dominant_orientation_strength"].append(ds)
            texture_trends["curvature_index"].append(ci)
            texture_trends["symmetry_score"].append(ss)
            texture_trends["composition_score"].append(cs)

    return years, color_trends, texture_trends


# ---------------- VISUALIZATION ----------------
def plot_trends(years, color_trends=None, texture_trends=None, mode="both"):
    OUTPUT_DIR = "../web/public/generirani_grafi/timeline/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.style.use("dark_background")

    def style_axis(ax):
        ax.set_facecolor("#050816")
        ax.grid(color="white", alpha=0.08, linewidth=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color((1, 1, 1, 0.1))
        ax.spines["bottom"].set_color((1, 1, 1, 0.1))
        ax.tick_params(colors="white", labelsize=10)

    def smooth(values, window=3):
        s = pd.Series(values, dtype=float)
        return s.rolling(window=window, center=True, min_periods=1).mean()

    def has_real_data(series_keys, trends):
        for key in series_keys:
            vals = np.array(trends.get(key, []), dtype=float)
            if vals.size > 0 and not np.all(np.isnan(vals)):
                return True
        return False

    def save_group(title, metrics, trends, filename, ylabel="Metric Value"):
        keys = [key for key, _ in metrics]
        if not has_real_data(keys, trends):
            print(f"Skipping {filename} because all values are NaN")
            return

        fig, ax = plt.subplots(figsize=(15, 8.5))
        fig.patch.set_facecolor("#050816")
        style_axis(ax)

        for key, label in metrics:
            vals = smooth(trends[key])
            ax.plot(years, vals, linewidth=3, alpha=0.92, label=label)
            ax.scatter(years, vals, s=22, alpha=0.65)

        ax.set_title(title, fontsize=24, pad=20, color="white")
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)

        legend = ax.legend(frameon=False, fontsize=12)
        for text in legend.get_texts():
            text.set_color("white")

        plt.tight_layout()
        plt.savefig(
            f"{OUTPUT_DIR}/{filename}",
            dpi=300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor()
        )
        plt.close()

    if mode in ("color", "both") and color_trends is not None:
        save_group(
            "Temporal Colour Evolution",
            [
                ("brightness", "Brightness"),
                ("warmth", "Warmth"),
                ("saturation", "Saturation"),
                ("entropy", "Complexity"),
            ],
            color_trends,
            "color_trends.png"
        )

    if mode in ("edge", "both") and texture_trends is not None:
        texture_groups = [
            {
                "title": "Texture Density",
                "metrics": [
                    ("mean_gradient", "Mean Gradient"),
                    ("edge_density", "Edge Density"),
                    ("laplacian_variance", "Laplacian Variance"),
                ],
                "file": "texture_density.png"
            },
            {
                "title": "Line Structure",
                "metrics": [
                    ("num_lines", "Number of Lines"),
                    ("mean_line_length", "Line Length"),
                    ("line_support_ratio", "Line Support"),
                ],
                "file": "line_structure.png"
            },
            {
                "title": "Curves & Orientation",
                "metrics": [
                    ("curve_edge_ratio", "Curve Ratio"),
                    ("orientation_entropy", "Orientation Entropy"),
                    ("dominant_orientation_strength", "Orientation Strength"),
                    ("curvature_index", "Curvature"),
                ],
                "file": "curve_structure.png"
            },
            {
                "title": "Symmetry & Composition",
                "metrics": [
                    ("symmetry_score", "Symmetry"),
                    ("composition_score", "Composition"),
                ],
                "file": "symmetry_composition.png"
            }
        ]

        for group in texture_groups:
            save_group(group["title"], group["metrics"], texture_trends, group["file"])


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

    print("Generating graphs...")
    plot_trends(years, color_trends, texture_trends, mode=args.mode)
    print("Graphs saved to generated_graphs/")

    print("Done.")


if __name__ == "__main__":
    main()
