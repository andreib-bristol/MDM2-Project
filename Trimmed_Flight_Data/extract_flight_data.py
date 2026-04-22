import os
import re
import pandas as pd

folder_name = "hf4"  # pick flight folder
folder = f"pigeonflocks_trajectories/{folder_name}" #adjust path
output_file = f"{folder_name}_data.py"

# adjust start and end times 
latest_start = 2024400
earliest_end = 2132360

headers = [
    "t(centisec)",
    "X(m)",
    "Y(m)",
    "Z(m)",
    "dX/dt(m/s)",
    "dY/dt(m/s)",
    "dZ/dt(m/s)",
    "d^2X/dt^2(m/s^2)",
    "d^2Y/dt^2(m/s^2)",
    "d^2Z/dt^2(m/s^2)",
    "GPS signal",
]

result = {}

for file in sorted(os.listdir(folder)):
    m = re.match(rf"{re.escape(folder_name)}_([A-Za-z])\.txt$", file)
    if not m:
        continue

    letter = m.group(1)
    path = os.path.join(folder, file)

    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=headers,
        header=None
    )

    # drop any embedded header rows
    df = df[df["t(centisec)"] != "t(centisec)"]

    
    # convert to numeric
    for c in headers:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()

    # >>> keep only the overlapping time window (inclusive) <<<
    df = df[(df["t(centisec)"] >= latest_start) & (df["t(centisec)"] <= earliest_end)]

    df["t(centisec)"] = df["t(centisec)"] - df["t(centisec)"].min()

    # store per pigeon
    result[letter] = df.values.tolist()

with open(output_file, "w") as f:
    f.write("data = ")
    f.write(repr(result))

print("Saved:", output_file)
