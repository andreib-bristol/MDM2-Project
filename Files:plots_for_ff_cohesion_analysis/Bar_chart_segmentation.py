import numpy as np
import pandas as pd
import os
import importlib.util
import matplotlib.pyplot as plt

# ---- PARAMETERS ----
window = 296
threshold = 16.672860308808694
min_duration = int(485)

data_folder = "FF_Trimmed_Flight_Data_for_threshold"
centroid_folder = "FF_Centroids_for_threshold"


# ---- LOAD DATA ----
def load_data_module(path):
    spec = importlib.util.spec_from_file_location("data_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.data


# ---- LOAD CENTROIDS ----
def load_centroids(path):
    centroids = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip().replace("[", "").replace("]", "")
            parts = line.split(",")
            row = [float(p) for p in parts if p != ""]
            centroids.append(row)
    return np.array(centroids)


# ---- MEDIAN DISTANCE ----
def median_distance_per_timestep(data, centroids):
    birds = list(data.keys())
    T = len(data[birds[0]])

    med_dist = np.empty(T)

    for t in range(T):
        cx, cy, cz = centroids[t]

        dists = []
        for bird in birds:
            row = data[bird][t]
            x, y, z = row[1], row[2], row[3]

            d = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
            dists.append(d)

        med_dist[t] = np.nanmedian(dists)

    return med_dist


# ---- SEGMENT FUNCTION ----
def extract_segments(labels):
    segments = []
    start = 0

    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            segments.append((start, i))
            start = i

    segments.append((start, len(labels)))
    return segments


# ---- STORAGE ----
stable_per_flight = []
erratic_per_flight = []

stable_percentages = []
erratic_percentages = []
flight_labels = []


# ---- MAIN LOOP ----
for i in range(1, 12):

    print(f"\nProcessing ff{i}...")

    # LOAD
    data_path = os.path.join(data_folder, f"ff{i}_data.py")
    data = load_data_module(data_path)

    centroid_path = os.path.join(centroid_folder, f"centroids_{i}.txt")
    centroids = load_centroids(centroid_path)

    # SIGNAL
    med_dist = median_distance_per_timestep(data, centroids)

    # ROLLING VARIANCE
    rolling_var = pd.Series(med_dist).rolling(window).var().to_numpy()

    # REMOVE NaNs (alignment critical)
    valid = ~np.isnan(rolling_var)
    rolling_var = rolling_var[valid]

    # LABELS
    labels = rolling_var > threshold

    # SEGMENTS
    segments = extract_segments(labels)

    stable_vars = []
    erratic_vars = []

    stable_time = 0
    erratic_time = 0
    total_time = len(rolling_var)

    for start, end in segments:
        length = end - start

        segment_var = rolling_var[start:end]
        seg_var = np.nanmean(segment_var)

        if labels[start] == False:
            if length >= min_duration:
                stable_vars.append(seg_var)
                stable_time += length
        else:
            erratic_vars.append(seg_var)
            erratic_time += length

    # ---- MEAN VARIANCES ----
    stable_mean = np.mean(stable_vars) if stable_vars else np.nan
    erratic_mean = np.mean(erratic_vars) if erratic_vars else np.nan

    stable_per_flight.append(stable_mean)
    erratic_per_flight.append(erratic_mean)

    # ---- NORMALISED TIME ----
    stable_frac = stable_time / total_time
    erratic_frac = erratic_time / total_time

    stable_percentages.append(stable_frac)
    erratic_percentages.append(erratic_frac)
    flight_labels.append(f"ff{i}")

    # PRINT
    print(f"ff{i} → Stable Var: {stable_mean:.3f}, Erratic Var: {erratic_mean:.3f}")
    print(f"      → Stable: {stable_frac:.2%}, Erratic: {erratic_frac:.2%}")


# ---- FINAL SUMMARY ----
print("\n===== FINAL RESULTS =====")

print("\nStable variance per flight:")
print(stable_per_flight)

print("\nErratic variance per flight:")
print(erratic_per_flight)

print("\nMean stable variance:", np.nanmean(stable_per_flight))
print("Mean erratic variance:", np.nanmean(erratic_per_flight))

print("\nMean % stable:", np.mean(stable_percentages))
print("Mean % erratic:", np.mean(erratic_percentages))

# ---- BAR CHART (FINAL VERSION) ----
x = np.arange(len(flight_labels))
width = 0.35

plt.figure(figsize=(12, 6))

# Use SAME colours as segmentation plots
stable_color = '#2ca02c'   # green
erratic_color = '#d62728'  # red

plt.bar(x - width/2, stable_percentages, width,
        label="Stable (Low Variance)", color=stable_color)

plt.bar(x + width/2, erratic_percentages, width,
        label="Erratic (High Variance)", color=erratic_color)

# ---- LABELS ----
plt.xlabel("Flight", fontsize=12)
plt.ylabel("Proportion of Flight Duration", fontsize=12)

plt.title("Time Spent in Stable and Erratic Behaviour Across Pigeon Flights",
          fontsize=14, fontweight='bold')

plt.xticks(x, flight_labels)
plt.ylim(0, 1)

# ---- VALUE LABELS ----
for i in range(len(x)):
    plt.text(x[i] - width/2, stable_percentages[i] + 0.02,
             f"{stable_percentages[i]*100:.0f}%",
             ha='center', fontsize=10)

    plt.text(x[i] + width/2, erratic_percentages[i] + 0.02,
             f"{erratic_percentages[i]*100:.0f}%",
             ha='center', fontsize=10)

# ---- LEGEND ----
plt.legend()

# ---- GRID (clean + subtle) ----
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()

# ---- SAVE FIGURE ----
plt.savefig("figure_cohesion_regimes_across_flights.png",
            dpi=300, bbox_inches='tight')

plt.show()