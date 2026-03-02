import pandas as pd
import numpy as np


file1 = "pigeonflocks_trajectories/ff1/ff1_A.txt"
file2 = "pigeonflocks_trajectories/ff1/ff1_F.txt"


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


dots = np.sum(u1 * u2, axis=1)
C0 = np.nanmean(dots)

print("C_ij(0) =", C0)
print("first 5 dot products:", dots[:5])

