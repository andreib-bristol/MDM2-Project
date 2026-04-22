import numpy as np
import matplotlib.pyplot as plt

# ---- choose your inputs here ----
import Trimmed_Flight_Data.ff3_data   # change to ff2_data, ff3_data, etc.
CENTROIDS_TXT = "Centroids/centroids_3.txt"       # centroids for the same flight
DT_SECONDS = 0.2                      # 20 centiseconds
# ---------------------------------

data = Trimmed_Flight_Data.ff3_data.data


def load_centroids(path):
    # Each line: cx, cy, cz
    centroids = np.loadtxt(path, delimiter=",")
    return np.atleast_2d(centroids)


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

        # Average across pigeons, ignoring NaNs
        avg_dist[t] = np.nanmedian(dists)

    return avg_dist


centroids = load_centroids(CENTROIDS_TXT)
avg_distances = average_distance_per_timestep(data, centroids)

# Build time axis: 0, 0.2, 0.4, ...
T = len(avg_distances)
time_seconds = np.arange(T) * DT_SECONDS

plt.figure()
plt.plot(time_seconds, avg_distances)
plt.xlabel("Time (s)")
plt.ylabel("Average distance to centroid (m)")
plt.title("Flock spread over time (avg distance to centroid)")
plt.show()
