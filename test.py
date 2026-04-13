import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levene

import Trimmed_Flight_Data.ff1_data
data = Trimmed_Flight_Data.ff1_data.data

rows = []

for bird_id, records in data.items():
    for row in records:
        rows.append({
            "bird_id": bird_id,
            "t": row[0],
            "x": row[1],
            "y": row[2],
            "z": row[3],
            "vx": row[4],
            "vy": row[5],
            "vz": row[6],
            "ax": row[7],
            "ay": row[8],
            "az": row[9],
            "gps": row[10]
        })

df = pd.DataFrame(rows)

df["speed"] = np.sqrt(df["vx"]**2 + df["vy"]**2 + df["vz"]**2)
df["acc_mag"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)



def levenefunc(df, comparison, birds):
    groups = [group[comparison].dropna().values for _, group in df.groupby(birds)]

    stat, p = levene(*groups)
    
    return stat, p

def test_speed_variance_birds(df):
    stat, p = levenefunc(df, "speed", "bird_id")
    
    print("Speed variance across birds")
    print(f"Statistic: {stat:.4f}, p-value: {p:.4e}\n")
    
    return stat, p

def test_acc_variance_birds(df):
    stat, p = levenefunc(df, "acc_mag", "bird_id")
    
    print("Acceleration variance across birds")
    print(f"Statistic: {stat:.4f}, p-value: {p:.4e}\n")
    
    return stat, p

test_speed_variance_birds(df)
test_acc_variance_birds(df)