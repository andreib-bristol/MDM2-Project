import numpy as np
import matplotlib.pyplot as plt

# ---- INPUTS ----
import Trimmed_Flight_Data.hf3_data  # change if needed
CENTROIDS_TXT = "Centroids/centroids_h3.txt"
DT_SECONDS = 0.2

# Choose pigeons to highlight/remove
SELECTED_BIRDS = ["L", "G"]  # change as needed
# ----------------

data = Trimmed_Flight_Data.hf3_data.data


# ---------- LOAD CENTROIDS ----------
def load_centroids(path):
    centroids = np.loadtxt(path, delimiter=",")
    return np.atleast_2d(centroids)


# ---------- DISTANCE FUNCTIONS ----------
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


def individual_distances_per_timestep(data, centroids):
    birds = list(data.keys())
    T = len(data[birds[0]])

    bird_distances = {bird: np.empty(T, dtype=float) for bird in birds}

    for t in range(T):
        cx, cy, cz = centroids[t]

        for bird in birds:
            row = data[bird][t]
            x, y, z = row[1], row[2], row[3]

            d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
            bird_distances[bird][t] = d

    return bird_distances


# ---------- NEW: RECOMPUTE CENTROIDS ----------
def centroids_excluding_birds(data, exclude_birds):
    birds = [b for b in data.keys() if b not in exclude_birds]
    T = len(data[birds[0]])

    centroids = np.empty((T, 3), dtype=float)

    for t in range(T):
        xs, ys, zs = [], [], []

        for bird in birds:
            row = data[bird][t]
            x, y, z = row[1], row[2], row[3]

            xs.append(x)
            ys.append(y)
            zs.append(z)

        centroids[t, 0] = np.nanmean(xs)
        centroids[t, 1] = np.nanmean(ys)
        centroids[t, 2] = np.nanmean(zs)

    return centroids


# ---------- MAIN ----------
centroids_full = load_centroids(CENTROIDS_TXT)

# Original metrics
avg_all = average_distance_per_timestep(data, centroids_full)
bird_distances = individual_distances_per_timestep(data, centroids_full)

# Recomputed centroids (excluding selected birds)
centroids_reduced = centroids_excluding_birds(data, SELECTED_BIRDS)

# New average based on reduced flock
avg_reduced = average_distance_per_timestep(data, centroids_reduced)

# Time axis
T = len(avg_all)
time_seconds = np.arange(T) * DT_SECONDS

# ---------- PLOTTING ----------
plt.figure()

# Plot selected pigeons
birds = list(bird_distances.keys())
cmap = plt.cm.get_cmap('tab20', len(birds))
for i, bird in enumerate(SELECTED_BIRDS):
    if bird in bird_distances:
        plt.plot(
            time_seconds,
            bird_distances[bird],
            color=cmap(birds.index(bird)),
            linewidth=2,
            label=f"Bird {bird}"
        )

# Plot original average
plt.plot(
    time_seconds,
    avg_all,
    linewidth=2.5,
    color="black",
    label="Average (all birds)"
)

# Plot reduced average
plt.plot(
    time_seconds,
    avg_reduced,
    color="green",
    linewidth=2.5,
    label="Average (excluding " f"{bird}"")"
)

plt.xlabel("Time (s)")
plt.ylabel("Distance to Centroid (m)")
plt.title("Distance to Flock Centroid Over Time (Homing Flight 4)")

plt.legend()
plt.tight_layout()
plt.show()


def plot_birds_eye_view(data, x_limits):
    plt.figure()

    birds = list(data.keys())
    cmap = plt.cm.get_cmap('tab20', len(birds))

    all_x, all_y = [], []

    # Store final positions for labeling
    end_positions = []

    for i, bird in enumerate(birds):
        xs = [row[1] for row in data[bird]]
        ys = [row[2] for row in data[bird]]

        all_x.extend(xs)
        all_y.extend(ys)

        color = cmap(i)

        lw = 2.5
        alpha = 1.0

        plt.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, label=f"Bird {bird}")

        # Save endpoint for label placement
        end_positions.append((bird, xs[-1], ys[-1], color))

    # ---- AXIS LIMITS ----
    y_min, y_max = min(all_y), max(all_y)
    y_pad = 0.05 * (y_max - y_min)

    plt.xlim(x_limits)
    plt.ylim(y_min - y_pad, y_max + y_pad)

    centroids = load_centroids(CENTROIDS_TXT)
    plt.plot(centroids_reduced[:,0], centroids_reduced[:,1], color='black', linewidth=2, label='Centroid \n(recomputed)')
    plt.plot(centroids_reduced[:,0][0], centroids_reduced[:,1][0], marker='X', color='red', markersize=10, label='Start')
    plt.plot(centroids_reduced[:,0][-1], centroids_reduced[:,1][-1], marker='X', color='blue', markersize=10, label='End')

    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.title("Birds Eye View of Homing Flight 4")
    plt.legend(loc='upper right', bbox_to_anchor=(1.7, 1))
    #plt.axis('equal')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

plot_birds_eye_view(data, x_limits=(0, 10000))

plt.figure()
plt.plot(time_seconds, avg_reduced, color='green', linewidth=2.5, label="Average (excluding selected birds)")
plt.xlabel("Time (s)")
plt.ylabel("Distance to Centroid (m)")
plt.title("Distance to Flock Centroid Over Time (Homing Flight 4)")
plt.legend()
plt.tight_layout()
plt.show()