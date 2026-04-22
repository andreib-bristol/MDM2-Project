import numpy as np

nan = np.nan   # allows the dataset to load correctly

import Trimmed_Flight_Data.hf4_data # adjust these for specific flight file

data = Trimmed_Flight_Data.hf4_data.data # adjust these for specific flight file


def compute_centroids(data):

    birds = list(data.keys())
    num_timesteps = len(data[birds[0]])

    centroids = []

    for t in range(num_timesteps):

        xs = []
        ys = []
        zs = []

        for bird in birds:

            row = data[bird][t]

            x = row[1]
            y = row[2]
            z = row[3]

            xs.append(x)
            ys.append(y)
            zs.append(z)

        # ignore NaN values when computing centroid
        cx = np.nanmean(xs)
        cy = np.nanmean(ys)
        cz = np.nanmean(zs)

        centroids.append([cx, cy, cz])

    return centroids


centroids = compute_centroids(data)

with open("centroids_2.txt", "w") as f:
    for c in centroids:
        f.write(f"{c[0]}, {c[1]}, {c[2]}\n")

print("Centroids saved to centroids.txt")
