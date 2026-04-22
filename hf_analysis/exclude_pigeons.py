import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---- INPUTS ----
import Trimmed_Flight_Data.hf4_data  # change if needed
CENTROIDS_TXT = "Centroids/centroids_h4.txt"
DT_SECONDS = 0.2

# Choose pigeons to highlight/remove
SELECTED_BIRDS = ["G"]  # change as needed
# ----------------

data = Trimmed_Flight_Data.hf4_data.data


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

plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})

# ---- LINE / AXIS STYLING ----
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6


target_bird  = "G"       # bird to highlight with a moving marker
step         = 5         # animate every Nth timestep (increase to speed up)
interval_ms  = 5       # milliseconds between frames
x_limits     = (0, 10000)

def animate_birds_eye_view(data, centroids_reduced, x_limits):

    birds = list(data.keys())
    cmap  = plt.cm.get_cmap('tab20', len(birds))

    # --- extract positions (same as your existing code) ---
    bird_xs = {}
    bird_ys = {}
    all_x, all_y = [], []

    for bird in birds:
        xs = [row[1] for row in data[bird]]
        ys = [row[2] for row in data[bird]]
        bird_xs[bird] = np.array(xs, dtype=float)
        bird_ys[bird] = np.array(ys, dtype=float)
        all_x.extend(xs)
        all_y.extend(ys)

    cx = centroids_reduced[:, 0]
    cy = centroids_reduced[:, 1]

    T      = len(bird_xs[birds[0]])
    frames = list(range(0, T, step))

    # --- figure ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal', adjustable='box')

    y_min, y_max = min(all_y), max(all_y)
    y_pad = 0.05 * (y_max - y_min)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_title("Birds Eye View of Homing Flight 4")

    # ---- static full-path lines (faint background, same as your plot) ----
    for i, bird in enumerate(birds):
        color = cmap(i)
        lw    = 2.5
        alpha = 0.8 
        if bird == target_bird:
            ax.plot(bird_xs[bird], bird_ys[bird],
                    color=color, linewidth=lw, alpha=alpha,
                    label=f"Bird {bird}", zorder=1)

    # centroid full path (faint)
    ax.plot(cx, cy, color='black', linewidth=2.0, alpha=1,
            label='Centroid \n(Recalculated)', zorder=1)

    # start / end markers (same as your plot)
    ax.plot(cx[0],  cy[0],  marker='X', color='red',  markersize=10,
            label='Start', zorder=5)
    ax.plot(cx[-1], cy[-1], marker='X', color='blue', markersize=10,
            label='End',   zorder=5)

    bird_G_color   = cmap(birds.index(target_bird))
    centroid_color = 'black'

    # ---- moving markers ----
    marker_G,    = ax.plot([], [], 'o', color=bird_G_color,   markersize=10,
                           zorder=4, )
    marker_cent, = ax.plot([], [], 'o', color=centroid_color, markersize=12,
                           zorder=4, )

    # labels that follow the markers
    label_G    = ax.annotate(f"Bird {target_bird}",
                             xy=(0, 0), xytext=(8, 8),
                             textcoords='offset points',
                             fontsize=9, color=bird_G_color,
                             fontweight='bold', zorder=5)
    label_cent = ax.annotate("Centroid",
                             xy=(0, 0), xytext=(8, 8),
                             textcoords='offset points',
                             fontsize=9, color='black',
                             fontweight='bold', zorder=5)

    # time stamp in corner
    time_text = ax.text(0.04, 0.98, '', transform=ax.transAxes,
                        fontsize=14, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.35))
    plt.tight_layout()

    # ---- update function called each frame ----
    def update(frame_idx):
        t = frames[frame_idx]

        # moving markers at current position
        marker_G.set_data([bird_xs[target_bird][t]],
                          [bird_ys[target_bird][t]])
        marker_cent.set_data([cx[t]], [cy[t]])

        # move labels with markers
        label_G.xy    = (bird_xs[target_bird][t], bird_ys[target_bird][t])
        label_cent.xy = (cx[t], cy[t])

        # time stamp
        time_text.set_text(f"t = {t * 0.2:.1f} s")

        return (marker_G, marker_cent,
                label_G, label_cent, time_text)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval_ms,
        blit=True
    )

    return ani

ani = animate_birds_eye_view(data, centroids_reduced, x_limits)

ani.save("flight4quick_animation.gif", writer="pillow", fps=40)
