import numpy as np
import matplotlib.pyplot as plt

# ---- choose your inputs here ----
import Trimmed_Flight_Data.hf4_data   # change to ff2_data, ff3_data, etc.
CENTROIDS_TXT = "Centroids/centroids_h4.txt"       # centroids for the same flight
DT_SECONDS = 0.2                      # 20 centiseconds
# ---------------------------------

data = Trimmed_Flight_Data.hf4_data.data
birds = list(data.keys())

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

def individual_distances_per_timestep(data, centroids):
    birds = list(data.keys())
    T = len(data[birds[0]])

    # Dictionary: bird -> distance array over time
    bird_distances = {bird: np.empty(T, dtype=float) for bird in birds}

    for t in range(T):
        cx, cy, cz = centroids[t]

        for bird in birds:
            row = data[bird][t]
            x, y, z = row[1], row[2], row[3]

            d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
            bird_distances[bird][t] = d

    return bird_distances

centroids = load_centroids(CENTROIDS_TXT)

avg_distances = average_distance_per_timestep(data, centroids)
bird_distances = individual_distances_per_timestep(data, centroids)

# Time axis
T = len(avg_distances)
time_seconds = np.arange(T) * DT_SECONDS

plt.figure()

birds = list(bird_distances.keys())

# Create a colormap with enough distinct colours
cmap = plt.cm.get_cmap('tab20', len(birds))

# Plot each pigeon with its own colour + label
for i, bird in enumerate(birds):
    plt.plot(
        time_seconds,
        bird_distances[bird],
        color=cmap(i),
        linewidth=2,
        label=f"Bird {bird}"
    )

# Plot average (make it stand out clearly)
plt.plot(
    time_seconds,
    avg_distances,
    color='black',
    linewidth=2.5,
    label="Average"
)

plt.xlabel("Time (s)")
plt.ylabel("Distance to Centroid (m)")
plt.title("Distance to Flock Centroid Over Time (Homing Flight 4)")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # puts legend outside
plt.tight_layout()

plt.show()
