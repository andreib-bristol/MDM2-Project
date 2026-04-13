import numpy as np


def load_flight(data: dict) -> tuple:
    """
    Convert raw pigeon flight data into arrays ready for compute_delay_matrices.

    The input dict is produced by the data extraction script (generate_all_data.py).
    Each bird's rows follow the column order:
        [t(centisec), X, Y, Z, dX/dt, dY/dt, dZ/dt, d2X/dt2, d2Y/dt2, d2Z/dt2, GPS signal]
        index: 0       1  2  3    4      5      6       7         8         9        10

    Birds are truncated to the shortest array length to ensure consistent shape
    across all birds, handling occasional missing rows within the time window.

    Parameters
    ----------
    data : dict
        bird_id -> list of rows, as produced by generate_all_data.py

    Returns
    -------
    t_seconds  : (T,) time array in seconds
    pos_xy     : dict bird_id -> (T, 2) X, Y positions in metres
    vel_xy     : dict bird_id -> (T, 2) X, Y velocities in m/s
    valid_mask : dict bird_id -> (T,) bool, True where GPS fix is real (not interpolated)
    """
    bird_ids = sorted(data.keys())

    lengths = {bird: len(rows) for bird, rows in data.items()}
    min_len = min(lengths.values())
    max_len = max(lengths.values())

    if max_len - min_len > 10:
        print(f"Warning: bird array lengths differ by {max_len - min_len} "
              f"rows — truncating all to {min_len}")

    first = np.array(data[bird_ids[0]])[:min_len]
    t_seconds = first[:, 0] / 100.0  # centiseconds -> seconds

    pos_xy     = {}
    vel_xy     = {}
    valid_mask = {}

    for bird, rows in data.items():
        arr = np.array(rows)[:min_len]

        pos_xy[bird]     = arr[:, 1:3]   # X, Y columns
        vel_xy[bird]     = arr[:, 4:6]   # dX/dt, dY/dt columns
        gps_signal = arr[:, 10].astype(float)
        window = np.ones(5)  # ±0.4s = 2 steps each side = 5 point window
        fix_counts = np.convolve(gps_signal, window, mode='same')
        valid_mask[bird] = fix_counts >= 2  # GPS signal: 1=real, 0=interpolated

        if valid_mask[bird].mean() < 0.5:
            print(f"Warning: bird {bird} has only "
                  f"{valid_mask[bird].mean():.0%} real GPS fixes — "
                  f"results may be unreliable")

    return t_seconds, pos_xy, vel_xy, valid_mask