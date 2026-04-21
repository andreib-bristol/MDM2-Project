import numpy as np
import pandas as pd
import os
import importlib.util

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

    # REMOVE NaNs FIRST (important for alignment)
    valid = ~np.isnan(rolling_var)
    rolling_var = rolling_var[valid]
    med_dist = med_dist[valid]

    # LABELS
    labels = rolling_var > threshold

    # SEGMENTS
    segments = extract_segments(labels)

    stable_vars = []
    erratic_vars = []

    for start, end in segments:
        length = end - start

        # Use rolling variance for measurement (KEY FIX)
        segment_var = rolling_var[start:end]
        seg_var = np.nanmean(segment_var)

        if labels[start] == False:
            # LOW VARIANCE
            if length >= min_duration:
                stable_vars.append(seg_var)
            # else: ignore short low-variance

        else:
            # HIGH VARIANCE → ALWAYS ERRATIC
            erratic_vars.append(seg_var)

    # STORE
    stable_mean = np.mean(stable_vars) if stable_vars else np.nan
    erratic_mean = np.mean(erratic_vars) if erratic_vars else np.nan

    stable_per_flight.append(stable_mean)
    erratic_per_flight.append(erratic_mean)

    print(f"ff{i} → Stable: {stable_mean:.3f}, Erratic: {erratic_mean:.3f}")


# ---- FINAL OUTPUT ----
print("\n===== FINAL RESULTS =====")

print("Stable per flight:")
print(stable_per_flight)

print("\nErratic per flight:")
print(erratic_per_flight)

print("\nOverall Means:")
print(f"Mean stable variance: {np.nanmean(stable_per_flight):.3f}")
print(f"Mean erratic variance: {np.nanmean(erratic_per_flight):.3f}")