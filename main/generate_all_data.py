import os
import re
import pandas as pd


TRAJECTORIES_DIR = "../pigeonflocks_trajectories"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

HEADERS = [
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


def process_folder(folder_name: str) -> None:
    """
    Extract and trim trajectory data for a single flight folder.

    Performs two passes over the raw .txt files:
      1. Find the overlapping time window across all birds in the flight
         (latest start time to earliest end time).
      2. Extract each bird's data within that window, reset time to zero,
         and save the result as a Python dict literal to a _data.py file.

    The output file can be imported directly:
        from data.ff1_data import data

    Skips the folder if the output file already exists, if no valid bird
    files are found, or if there is no overlapping time window.

    Parameters
    ----------
    folder_name : str
        Name of the flight folder, e.g. 'hf4' or 'ff1'.
        Must match the prefix of the bird .txt files inside it.
    """
    folder = os.path.join(TRAJECTORIES_DIR, folder_name)
    output_file = os.path.join(OUTPUT_DIR, f"{folder_name}_data.py")

    if os.path.exists(output_file):
        print(f"  Skipping {folder_name} — already exists")
        return

    result = {}
    time_ranges = []

    # First pass: find overlapping time window across all birds
    for file in sorted(os.listdir(folder)):
        m = re.match(rf"{re.escape(folder_name)}_([A-Za-z])\.txt$", file)
        if not m:
            continue

        path = os.path.join(folder, file)
        df = pd.read_csv(path, sep=r"\s+", comment="#", names=HEADERS, header=None)
        df = df[df["t(centisec)"] != "t(centisec)"]
        for c in HEADERS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()

        time_ranges.append((df["t(centisec)"].min(), df["t(centisec)"].max()))

    if not time_ranges:
        print(f"  Skipping {folder_name} — no valid files found")
        return

    latest_start = max(t[0] for t in time_ranges)
    earliest_end = min(t[1] for t in time_ranges)

    if latest_start >= earliest_end:
        print(f"  Skipping {folder_name} — no overlapping time window")
        return

    # Second pass: extract data within the overlapping window
    for file in sorted(os.listdir(folder)):
        m = re.match(rf"{re.escape(folder_name)}_([A-Za-z])\.txt$", file)
        if not m:
            continue

        letter = m.group(1)
        path = os.path.join(folder, file)

        df = pd.read_csv(path, sep=r"\s+", comment="#", names=HEADERS, header=None)
        df = df[df["t(centisec)"] != "t(centisec)"]
        for c in HEADERS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()

        df = df[(df["t(centisec)"] >= latest_start) & (df["t(centisec)"] <= earliest_end)]
        df["t(centisec)"] = df["t(centisec)"] - df["t(centisec)"].min()

        result[letter] = df.values.tolist()

    with open(output_file, "w") as f:
        f.write("data = ")
        f.write(repr(result))

    print(f"  Saved: {output_file}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    flights = sorted([
        f for f in os.listdir(TRAJECTORIES_DIR)
        if os.path.isdir(os.path.join(TRAJECTORIES_DIR, f))
        and re.match(r"(ff|hf)\d+$", f)
    ])

    print(f"Found {len(flights)} flight folders: {flights}\n")

    for flight in flights:
        print(f"Processing {flight}...")
        try:
            process_folder(flight)
        except Exception as e:
            print(f"  ERROR in {flight}: {e}")

    print("\nDone.")