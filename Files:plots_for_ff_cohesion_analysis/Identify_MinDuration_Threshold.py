import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans

# -------------------------------
# SETTINGS
# -------------------------------
window = 296  

data_folder = "FF_Trimmed_Flight_Data_for_threshold"
centroid_folder = "FF_Centroids_for_threshold"


# -------------------------------
# LOAD CENTROIDS (ROBUST)
# -------------------------------
def load_centroids(path):
    centroids = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip().replace("[", "").replace("]", "")
            parts = line.split(",")
            row = [float(p) for p in parts if p != ""]
            centroids.append(row)
    return np.array(centroids)


# -------------------------------
# YOUR FUNCTION
# -------------------------------
def average_distance_per_timestep(data, centroids):
    birds = list(data.keys())
    T = len(data[birds[0]])

    avg_dist = np.empty(T, dtype=float)

    for t in range(T):
        cx, cy, cz = centroids[t]

        dists = []
        for bird in birds:
            row = data[bird][t]
            x, y, z = row[1], row[2], row[3]

            d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
            dists.append(d)

        avg_dist[t] = np.nanmedian(dists)

    return avg_dist


# -------------------------------
# STEP 1: COLLECT ALL VARIANCES
# -------------------------------
all_variances = []

for i in range(1, 12):

    print(f"Processing ff{i}...")

    # LOAD DATA
    data_path = os.path.join(data_folder, f"ff{i}_data.py")
    namespace = {}
    with open(data_path) as f:
        exec(f.read(), namespace)
    data = namespace["data"]

    # LOAD CENTROIDS
    centroid_path = os.path.join(centroid_folder, f"centroids_{i}.txt")
    centroids = load_centroids(centroid_path)

    # COMPUTE SIGNAL
    avg_dist = average_distance_per_timestep(data, centroids)
    rolling_var = pd.Series(avg_dist).rolling(window).var()

    clean_var = rolling_var.dropna().values
    all_variances.append(clean_var)


# COMBINE ALL
all_variances = np.concatenate(all_variances)

print("\nTotal variance samples:", len(all_variances))


# -------------------------------
# STEP 2: FIND VARIANCE THRESHOLD (KMEANS)
# -------------------------------
# log transform helps clustering
var_log = np.log1p(all_variances).reshape(-1, 1)

kmeans_var = KMeans(n_clusters=2, random_state=0).fit(var_log)

centers_log = sorted(kmeans_var.cluster_centers_.flatten())
centers = np.expm1(centers_log)

variance_threshold = np.mean(centers)

print("\nVariance cluster centers:", centers)
print("FINAL VARIANCE THRESHOLD:", variance_threshold)


# -------------------------------
# STEP 3: COLLECT SEGMENT LENGTHS
# -------------------------------
all_segment_lengths = []

for i in range(1, 12):

    print(f"Segmenting ff{i}...")

    # LOAD DATA
    data_path = os.path.join(data_folder, f"ff{i}_data.py")
    namespace = {}
    with open(data_path) as f:
        exec(f.read(), namespace)
    data = namespace["data"]

    # LOAD CENTROIDS
    centroid_path = os.path.join(centroid_folder, f"centroids_{i}.txt")
    centroids = load_centroids(centroid_path)

    # COMPUTE SIGNAL
    avg_dist = average_distance_per_timestep(data, centroids)
    rolling_var = pd.Series(avg_dist).rolling(window).var()

    # CLASSIFY
    labels = (rolling_var > variance_threshold).astype(int).dropna().values

    # EXTRACT SEGMENTS
    current_label = labels[0]
    length = 1

    for t in range(1, len(labels)):
        if labels[t] == current_label:
            length += 1
        else:
            all_segment_lengths.append(length)
            current_label = labels[t]
            length = 1

    all_segment_lengths.append(length)


all_segment_lengths = np.array(all_segment_lengths)

print("\nTotal segments:", len(all_segment_lengths))


# -------------------------------
# STEP 4: FIND MIN DURATION (KMEANS)
# -------------------------------
length_log = np.log1p(all_segment_lengths).reshape(-1, 1)

kmeans_len = KMeans(n_clusters=2, random_state=0).fit(length_log)

centers_log = sorted(kmeans_len.cluster_centers_.flatten())
centers = np.expm1(centers_log)

min_duration = np.mean(centers)

print("\nSegment length cluster centers:", centers)
print("FINAL MINIMUM DURATION:", min_duration)