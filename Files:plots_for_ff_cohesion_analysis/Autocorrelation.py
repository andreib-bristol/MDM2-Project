import numpy as np
import os

# ---- SETTINGS ----
dt = 0.2  # time step (seconds)

data_folder = "FF_Trimmed_Flight_Data_for_threshold"
centroid_folder = "FF_Centroids_for_threshold"


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


# ---- YOUR FUNCTION ----
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


# ---- AUTOCORRELATION ----
def autocorrelation(x):
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]


# ---- FIND DECORRELATION TIME USING 1/e ----
def find_decorrelation_time(avg_dist, dt):
    acf = autocorrelation(avg_dist)
    acf = acf / acf[0]  # normalize

    threshold = 1 / np.e  # ≈ 0.37 (theoretical)

    for lag in range(len(acf)):
        if acf[lag] < threshold:
            return lag * dt  # seconds

    return None


# ---- MAIN LOOP ----
decorrelation_times = []

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

    # COMPUTE AVG DISTANCE
    avg_dist = average_distance_per_timestep(data, centroids)

    # FIND DECORRELATION TIME
    tau = find_decorrelation_time(avg_dist, dt)

    if tau is not None:
        decorrelation_times.append(tau)
        print(f"ff{i}: decorrelation time = {tau:.2f} s")
    else:
        print(f"ff{i}: decorrelation time not found")


# ---- FINAL WINDOW ----
median_time = np.median(decorrelation_times)
window = int(round(median_time / dt))

print("\nAll decorrelation times:", decorrelation_times)
print(f"\nMedian decorrelation time: {median_time:.2f} s")
print(f"FINAL WINDOW (timesteps): {window}")