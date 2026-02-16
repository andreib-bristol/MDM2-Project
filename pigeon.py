import pandas as pd

file1 = "pigeonflocks_trajectories/ff1/ff1_A.txt"
file2 = "pigeonflocks_trajectories/ff1/ff1_F.txt"


data1 = pd.read_csv(file1, sep=r"\s+", skiprows=19, header=None)
data2 = pd.read_csv(file2, sep=r"\s+", skiprows=19, header=None)

columns = ["t","x", "y", "z","vx", "vy", "vz","ax", "ay", "az","gps"]

data1.columns = columns
data2.columns = columns

print(data1.head)