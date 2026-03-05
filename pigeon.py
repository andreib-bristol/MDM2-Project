import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file1 = "pigeonflocks_trajectories/ff1/ff1_A.txt"
file2 = "pigeonflocks_trajectories/ff1/ff1_G.txt"


data1 = pd.read_csv(file1, sep=r"\s+", skiprows=19, header=None)
data2 = pd.read_csv(file2, sep=r"\s+", skiprows=19, header=None)

columns = ["t","x", "y", "z","vx", "vy", "vz","ax", "ay", "az","gps"]

data1.columns = columns
data2.columns = columns

t_start = max(data1["t"].min(), data2["t"].min())
t_end   = min(data1["t"].max(), data2["t"].max())

d1 = data1[(data1["t"] >= t_start) & (data1["t"] <= t_end)].reset_index(drop=True)
d2 = data2[(data2["t"] >= t_start) & (data2["t"] <= t_end)].reset_index(drop=True)

#print(d1.shape, d2.shape)



def unit_heading(data):
    v = data[["vx", "vy", "vz"]].to_numpy()
    #print(v[:5])
    speed = np.sqrt(np.sum(v**2, axis=1))
    #print(speed[:5])
    speed[speed == 0] = np.nan
    u = v / speed[:, None]
    return u


u1 = unit_heading(d1)
u2 = unit_heading(d2)


#print(u1.shape, u2.shape)



def time_delay(u1, u2, t):
    n = len(u1)
    if t > 0:
        a = u1[0:n-t]
        b = u2[t:n]
    elif t < 0:
        t = -t
        a = u1[t:n]
        b = u2[0:n-t]
    else:
        a = u1
        b = u2
    dots = np.sum(a * b, axis=1)
    return np.nanmean(dots)


delays = list(range(-5, 5  ))
angles = [time_delay(u1, u2, L) for L in delays]

best_idx = int(np.nanargmax(angles))
best_delay = delays[best_idx]
best_angle = angles[best_idx]

print("best time delay:", best_delay, "samples")
print("best time delay:", best_delay * 0.2, "seconds")
print("best cos(theta):", best_angle)

dt = 0.2
delays = np.array(delays) * dt
angles = np.array(angles)

plt.figure()
plt.plot(delays, angles)
plt.xlabel("time delay")
plt.ylabel("cos(theta) values")
plt.grid()
plt.show()
