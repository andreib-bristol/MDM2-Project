import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levene
import importlib.util
import os

# ---- PARAMETERS (match your FF code exactly) ----
window = 296
threshold = 16.672860308808694
min_duration = int(485)

# ---- CHANGE THESE to point at your flight ----
DATA_PATH = "Trimmed_Flight_Data/ff1_data.py"
CENTROID_PATH = "Centroids/centroids_1.txt"  # update this


# ---- LOADERS ----
def load_data_module(path):
    spec = importlib.util.spec_from_file_location("data_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.data


def load_centroids(path):
    centroids = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip().replace("[", "").replace("]", "")
            parts = line.split(",")
            row = [float(p) for p in parts if p != ""]
            centroids.append(row)
    return np.array(centroids)


# ---- MEDIAN DISTANCE FROM CENTROID ----
def median_distance_per_timestep(data, centroids):
    birds = list(data.keys())
    T = len(data[birds[0]])
    med_dist = np.empty(T)
    for t in range(T):
        cx, cy, cz = centroids[t]
        dists = [
            np.sqrt((data[b][t][1] - cx)**2 +
                    (data[b][t][2] - cy)**2 +
                    (data[b][t][3] - cz)**2)
            for b in birds
        ]
        med_dist[t] = np.nanmedian(dists)
    return med_dist


# ---- FIND STABLE TIMESTEPS ----
def get_stable_timesteps(data, centroids):
    """
    Returns a boolean mask (length = number of valid rolling-var timesteps)
    and the offset needed to align it back to the original timestep indices.
    """
    med_dist = median_distance_per_timestep(data, centroids)

    rolling_var = pd.Series(med_dist).rolling(window).var().to_numpy()

    # Track which original indices survive the NaN drop
    valid_mask = ~np.isnan(rolling_var)
    valid_indices = np.where(valid_mask)[0]       # original timestep indices
    rolling_var = rolling_var[valid_mask]

    labels = rolling_var > threshold              # True = erratic

    # Walk segments, keep stable ones that meet min_duration
    stable_set = set()
    segments = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            segments.append((start, i))
            start = i
    segments.append((start, len(labels)))

    for seg_start, seg_end in segments:
        length = seg_end - seg_start
        if labels[seg_start] == False and length >= min_duration:
            # Map back to original timestep indices
            for pos in range(seg_start, seg_end):
                stable_set.add(int(valid_indices[pos]))

    return stable_set


# ---- BUILD DATAFRAME ----
def build_df(data):
    rows = []
    for bird_id, records in data.items():
        for row in records:
            rows.append({
                "bird_id": bird_id,
                "t":      row[0],
                "x":      row[1],  "y": row[2],  "z": row[3],
                "vx":     row[4],  "vy": row[5], "vz": row[6],
                "ax":     row[7],  "ay": row[8], "az": row[9],
                "gps":    row[10]
            })
    df = pd.DataFrame(rows)
    df["speed"]   = np.sqrt(df["vx"]**2 + df["vy"]**2 + df["vz"]**2)
    df["acc_mag"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
    return df


# ---- LEVENE HELPERS ----
def levenefunc(df, comparison, group_col):
    groups = [g[comparison].dropna().values for _, g in df.groupby(group_col)]
    stat, p = levene(*groups)
    return stat, p


def run_levene_tests(df, label=""):
    tag = f" [{label}]" if label else ""

    stat, p = levenefunc(df, "speed", "bird_id")
    print(f"Speed variance across birds{tag}")
    print(f"  Statistic: {stat:.4f},  p-value: {p:.4e}\n")

    stat, p = levenefunc(df, "acc_mag", "bird_id")
    print(f"Acceleration variance across birds{tag}")
    print(f"  Statistic: {stat:.4f},  p-value: {p:.4e}\n")

    print(f"Per-bird variance summary{tag}")
    print(df.groupby("bird_id")[["speed", "acc_mag"]].var())
    print()

def plot_variance_table(df_full, df_stable):
    
    full_var  = df_full.groupby("bird_id")[["speed", "acc_mag"]].var().round(3)
    stable_var = df_stable.groupby("bird_id")[["speed", "acc_mag"]].var().round(3)

    birds = full_var.index.tolist()

    col_labels = ["Bird", "Speed (full)", "Acc (full)", "Speed (stable)", "Acc (stable)"]
    table_data = []
    for bird in birds:
        table_data.append([
            bird,
            f"{full_var.loc[bird, 'speed']:.3f}",
            f"{full_var.loc[bird, 'acc_mag']:.3f}",
            f"{stable_var.loc[bird, 'speed']:.3f}",
            f"{stable_var.loc[bird, 'acc_mag']:.3f}",
        ])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.0)

    # ---- HEADER ROW styling ----
    for col in range(len(col_labels)):
        table[0, col].set_facecolor("#2c2c2c")
        table[0, col].set_text_props(color="white", fontweight="bold")

    # ---- COLOUR CELLS by value (full vs stable contrast) ----
    full_speed_max  = full_var["speed"].max()
    full_acc_max    = full_var["acc_mag"].max()
    stable_speed_max = stable_var["speed"].max()
    stable_acc_max   = stable_var["acc_mag"].max()

    for row_idx, bird in enumerate(birds, start=1):
        # Alternating row background
        base_color = "#f9f9f9" if row_idx % 2 == 0 else "white"
        for col in range(len(col_labels)):
            table[row_idx, col].set_facecolor(base_color)

        # Bird label column
        table[row_idx, 0].set_text_props(fontweight="bold")

        # Full flight columns — amber tint scaled by value
        for col_idx, (val, max_val) in enumerate([
            (full_var.loc[bird, "speed"],   full_speed_max),
            (full_var.loc[bird, "acc_mag"], full_acc_max),
        ], start=1):
            intensity = val / max_val
            r = 1.0
            g = 1.0 - 0.45 * intensity
            b = 0.6 - 0.55 * intensity
            table[row_idx, col_idx].set_facecolor((r, g, b))

        # Stable columns — green tint scaled by value
        for col_idx, (val, max_val) in enumerate([
            (stable_var.loc[bird, "speed"],   stable_speed_max),
            (stable_var.loc[bird, "acc_mag"], stable_acc_max),
        ], start=3):
            intensity = val / max_val
            r = 0.6 - 0.55 * intensity
            g = 0.85
            b = 0.6 - 0.55 * intensity
            table[row_idx, col_idx].set_facecolor((r, g, b))

    plt.title("Per-bird Variance: Full Flight vs Stable Segments",
              fontsize=13, fontweight="bold", pad=16)

    plt.tight_layout()
    plt.savefig("figure_variance_table.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_speed_boxplot(df_full):
    
    fig, ax = plt.subplots(figsize=(10, 5))

    bird_ids = sorted(df_full["bird_id"].unique())
    speed_data = [df_full[df_full["bird_id"] == b]["speed"].values for b in bird_ids]

    bp = ax.boxplot(
        speed_data,
        labels=bird_ids,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color="#555555", linewidth=1.2),
        capprops=dict(color="#555555", linewidth=1.2),
        flierprops=dict(marker="o", markersize=3,
                        markerfacecolor="#cccccc", markeredgecolor="#aaaaaa",
                        alpha=0.4),
        boxprops=dict(linewidth=1.2)
    )

    # ---- COLOUR boxes by variance (low = green, high = amber) ----
    variances = [df_full[df_full["bird_id"] == b]["speed"].var() for b in bird_ids]
    min_var = min(variances)
    max_var = max(variances)

    for patch, var in zip(bp["boxes"], variances):
        intensity = (var - min_var) / (max_var - min_var)
        r = 0.18 + 0.62 * intensity
        g = 0.55 - 0.13 * intensity
        b_val = 0.18 - 0.05 * intensity
        patch.set_facecolor((r, g, b_val))

    # ---- VARIANCE annotations above each box ----
    for i, (bird, var) in enumerate(zip(bird_ids, variances), start=1):
        ax.text(i, ax.get_ylim()[1] * 0.97,
                f"var={var:.1f}",
                ha="center", va="top",
                fontsize=9, color="#444444")

    ax.set_xlabel("Bird", fontsize=12)
    ax.set_ylabel("Speed (m/s)", fontsize=12)
    ax.set_title("Speed Distribution per Bird — Full Flight",
                 fontsize=13, fontweight="bold")

    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig("figure_speed_boxplot.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_variance_table_full(df_full):
    
    full_var = df_full.groupby("bird_id")[["speed", "acc_mag"]].var().round(3)
    birds = full_var.index.tolist()

    col_labels = ["Bird", "Speed variance", "Acceleration variance"]
    table_data = []
    for bird in birds:
        table_data.append([
            bird,
            f"{full_var.loc[bird, 'speed']:.3f}",
            f"{full_var.loc[bird, 'acc_mag']:.3f}",
        ])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.0)

    # ---- HEADER ROW ----
    for col in range(len(col_labels)):
        table[0, col].set_facecolor("#2c2c2c")
        table[0, col].set_text_props(color="white", fontweight="bold")

    # ---- COLOUR cells by value ----
    speed_max = full_var["speed"].max()
    acc_max   = full_var["acc_mag"].max()

    for row_idx, bird in enumerate(birds, start=1):
        base_color = "#f9f9f9" if row_idx % 2 == 0 else "white"
        table[row_idx, 0].set_facecolor(base_color)
        table[row_idx, 0].set_text_props(fontweight="bold")

        for col_idx, (val, max_val) in enumerate([
            (full_var.loc[bird, "speed"],   speed_max),
            (full_var.loc[bird, "acc_mag"], acc_max),
        ], start=1):
            intensity = val / max_val
            r = 1.0
            g = 1.0 - 0.45 * intensity
            b = 0.6 - 0.55 * intensity
            table[row_idx, col_idx].set_facecolor((r, g, b))

    plt.title("Per-bird variance - ff1",
              fontsize=13, fontweight="bold", pad=16)

    plt.tight_layout()
    plt.savefig("figure_variance_table_full.png", dpi=300, bbox_inches="tight")
    plt.show()


# ================================================================
# MAIN
# ================================================================

data      = load_data_module(DATA_PATH)
centroids = load_centroids(CENTROID_PATH)

df = build_df(data)

# --- Run on full flight first (baseline) ---
print("=" * 50)
print("FULL FLIGHT")
print("=" * 50)
run_levene_tests(df, label="full flight")

# --- Filter to stable timesteps only ---
stable_timesteps = get_stable_timesteps(data, centroids)

# The 't' column holds the original timestep index — filter on that
# If 't' is not a 0-based integer index, use the positional row index instead:
# df["_row"] = df.groupby("bird_id").cumcount()  and filter on that
df_stable = df[df["t"].isin(stable_timesteps)]

print(f"Stable timesteps retained: {len(stable_timesteps)} "
      f"/ {len(data[list(data.keys())[0]])} "
      f"({100*len(stable_timesteps)/len(data[list(data.keys())[0]]):.1f}%)\n")

print("=" * 50)
print("STABLE SEGMENTS ONLY")
print("=" * 50)
run_levene_tests(df_stable, label="stable only")

#plot_variance_table(df, df_stable)


#plot_speed_boxplot(df)

# ---- CALL IT ----
#plot_variance_table_full(df)

def plot_stat_callouts():

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    # ---- BACKGROUND PANELS ----
    # Full flight panel
    fig.add_axes([0.05, 0.1, 0.42, 0.8]).axis("off")
    left_panel = plt.axes([0.05, 0.1, 0.42, 0.8])
    left_panel.set_facecolor("#f5f5f5")
    for spine in left_panel.spines.values():
        spine.set_visible(False)
    left_panel.set_xticks([])
    left_panel.set_yticks([])

    # Stable panel
    right_panel = plt.axes([0.53, 0.1, 0.42, 0.8])
    right_panel.set_facecolor("#f5f5f5")
    for spine in right_panel.spines.values():
        spine.set_visible(False)
    right_panel.set_xticks([])
    right_panel.set_yticks([])

    # ---- COLUMN HEADERS ----
    fig.text(0.26, 0.92, "Full Flight",
             ha="center", fontsize=13, fontweight="bold", color="#2c2c2c")
    fig.text(0.74, 0.92, "Stable Only",
             ha="center", fontsize=13, fontweight="bold", color="#2c2c2c")

    # ---- DIVIDER ----
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.05, 0.95],
                              color="#cccccc", linewidth=1.2))

    # ---- FULL FLIGHT VALUES ----
    # W statistic
    fig.text(0.26, 0.68, "W",
             ha="center", fontsize=12, color="#888888")
    fig.text(0.26, 0.52, "1316",
             ha="center", fontsize=36, fontweight="bold", color="#BA7517")

    # eta squared
    fig.text(0.26, 0.35, "η²",
             ha="center", fontsize=12, color="#888888")
    fig.text(0.26, 0.18, "0.016",
             ha="center", fontsize=36, fontweight="bold", color="#BA7517")

    # ---- STABLE ONLY VALUES ----
    # W statistic
    fig.text(0.74, 0.68, "W",
             ha="center", fontsize=12, color="#888888")
    fig.text(0.74, 0.52, "67",
             ha="center", fontsize=36, fontweight="bold", color="#2ca02c")

    # eta squared
    fig.text(0.74, 0.35, "η²",
             ha="center", fontsize=12, color="#888888")
    fig.text(0.74, 0.18, "0.0014",
             ha="center", fontsize=36, fontweight="bold", color="#2ca02c")

    # ---- DROP ARROWS ----
    fig.text(0.50, 0.58, "↓ 95%",
             ha="center", fontsize=11, fontweight="bold",
             color="#d62728",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#d62728", linewidth=1.2))

    fig.text(0.50, 0.24, "↓ 90%",
             ha="center", fontsize=11, fontweight="bold",
             color="#d62728",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#d62728", linewidth=1.2))

    plt.savefig("figure_stat_callouts.png", dpi=300, bbox_inches="tight")
    plt.show()


# ---- CALL IT ----
plot_stat_callouts()