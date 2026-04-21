import os
import re
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

from loader import load_flight


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_data_module(path: str) -> dict:
    """Load a generated flight data module by file path."""
    spec = importlib.util.spec_from_file_location("flight_data", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "data")


def compute_order_parameter(
    vel_xy: dict,
    valid_mask: dict,
) -> tuple:
    """
    Compute the Vicsek order parameter phi(t) for a single flight.

    phi(t) = |mean unit velocity of all birds at time t|

    A value of 1 means all birds are perfectly aligned.
    A value of 0 means birds are pointing in completely random directions.

    Only timepoints where ALL birds have a valid GPS fix and non-zero
    speed are included. Timepoints where any bird's data is invalid
    are set to nan.

    Parameters
    ----------
    vel_xy     : dict bird_id -> (T, 2) velocity arrays
    valid_mask : dict bird_id -> (T,) bool validity masks

    Returns
    -------
    phi        : (T,) order parameter time series, nan where invalid
    mean_phi   : float, mean of valid phi values
    """
    bird_ids = sorted(vel_xy.keys())
    T = vel_xy[bird_ids[0]].shape[0]

    # Compute unit vectors and speed masks for each bird
    unit_vecs  = {}
    speed_mask = {}
    for bird in bird_ids:
        v = vel_xy[bird]
        speed = np.linalg.norm(v, axis=1)
        good = speed > 1e-6
        u = np.zeros_like(v)
        u[good] = v[good] / speed[good, np.newaxis]
        unit_vecs[bird]  = u
        speed_mask[bird] = good

    # At each timestep, all birds must be valid and moving
    all_valid = np.ones(T, dtype=bool)
    for bird in bird_ids:
        all_valid &= valid_mask[bird] & speed_mask[bird]

    # Stack unit vectors: shape (N, T, 2)
    U = np.stack([unit_vecs[b] for b in bird_ids], axis=0)

    # Mean unit vector across birds at each timestep: shape (T, 2)
    mean_u = U.mean(axis=0)

    # Order parameter is the magnitude of the mean unit vector
    phi = np.linalg.norm(mean_u, axis=1)

    # Mask invalid timesteps
    phi[~all_valid] = np.nan

    mean_phi = float(np.nanmean(phi))
    return phi, mean_phi


def plot_order_parameter(
    phi_dict: dict,
    t_dict: dict,
    output_path: str = "order_parameter.png",
) -> None:
    """
    Plot phi(t) time series for multiple flights on one figure.

    Parameters
    ----------
    phi_dict    : dict flight_name -> phi array
    t_dict      : dict flight_name -> t_seconds array
    output_path : filepath to save the figure
    """
    # Separate homing and free flights for colouring
    homing = {f: phi for f, phi in phi_dict.items() if f.startswith("hf")}
    free   = {f: phi for f, phi in phi_dict.items() if f.startswith("ff")}

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    for ax, flight_dict, title, color in zip(
        axes,
        [homing, free],
        ["Homing flights", "Free flights"],
        ["steelblue", "darkorange"],
    ):
        for flight_name, phi in flight_dict.items():
            t = t_dict[flight_name]
            # Trim t to match phi length if needed
            t = t[:len(phi)]
            mean_val = float(np.nanmean(phi))
            ax.plot(t, phi, alpha=0.7, label=f"{flight_name} (mean={mean_val:.3f})")

        ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8,
                   label="Quality threshold (0.5)")
        ax.set_ylabel("Order parameter φ(t)")
        ax.set_xlabel("Time (s)")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="lower right")

    plt.suptitle("Vicsek order parameter across flights\n"
                 "φ=1: perfect alignment, φ=0: random directions",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {output_path}")


def plot_mean_phi_vs_mean_cmax(
    mean_phi_dict: dict,
    mean_cmax_dict: dict,
    quality_threshold: float = 0.5,
    output_path: str = "phi_vs_cmax.png",
) -> None:
    """
    Scatter plot of mean phi against mean C_max per flight.

    All flights are plotted using phi. Flights that passed the C_max
    quality threshold are shown as filled circles, others as open circles.
    Homing flights are blue, free flights are orange.
    """
    from scipy.stats import pearsonr
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    fig, ax = plt.subplots(figsize=(9, 6))

    for flight in sorted(mean_phi_dict):
        phi_val  = mean_phi_dict[flight]
        cmax_val = mean_cmax_dict.get(flight, np.nan)
        is_homing  = flight.startswith("hf")
        passed     = (not np.isnan(cmax_val)) and (cmax_val >= quality_threshold)

        color      = "steelblue" if is_homing else "darkorange"
        marker     = "o"
        facecolor  = color if passed else "white"
        edgecolor  = color

        if not np.isnan(cmax_val):
            ax.scatter(phi_val, cmax_val,
                       c=[facecolor], edgecolors=edgecolor,
                       s=100, linewidths=1.5,
                       marker=marker, zorder=3)
            ax.annotate(flight, (phi_val, cmax_val),
                        textcoords="offset points",
                        xytext=(6, 4), fontsize=8)

    # Draw quality threshold lines
    ax.axhline(quality_threshold, color="gray", linestyle="--",
               linewidth=0.8, alpha=0.6, label=f"C_max threshold ({quality_threshold})")

    # Trend line using only flights with both values
    common = sorted(set(mean_phi_dict) & set(mean_cmax_dict))
    phi_vals  = np.array([mean_phi_dict[f]  for f in common])
    cmax_vals = np.array([mean_cmax_dict[f] for f in common])
    valid = ~(np.isnan(phi_vals) | np.isnan(cmax_vals))

    if valid.sum() >= 3:
        m, b = np.polyfit(phi_vals[valid], cmax_vals[valid], 1)
        x_line = np.linspace(phi_vals[valid].min(), phi_vals[valid].max(), 100)
        ax.plot(x_line, m * x_line + b, "k--", linewidth=1, alpha=0.4)
        r, p = pearsonr(phi_vals[valid], cmax_vals[valid])
        ax.set_title(f"Mean order parameter vs mean C_max (all flights)\n"
                     f"Pearson r={r:+.3f}, p={p:.3f}, n={valid.sum()} flights", fontsize=18)
    else:
        ax.set_title("Mean order parameter vs mean C_max", fontsize=18)

    # Legend
    legend_handles = [
        mpatches.Patch(color="steelblue",  label="Homing flights"),
        mpatches.Patch(color="darkorange", label="Free flights"),
        mlines.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor="gray", markeredgecolor="gray",
                      markersize=8, label="Passed C_max threshold"),
        mlines.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor="white", markeredgecolor="gray",
                      markersize=8, label="Below C_max threshold"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper left")

    ax.set_xlabel("Mean φ (order parameter)", fontsize=18)
    ax.set_ylabel("Mean C_max (directional correlation)", fontsize=18)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    from correlation import compute_delay_matrices

    data_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if re.match(r"(ff|hf)\d+_data\.py$", f)
    ])

    print(f"Found {len(data_files)} flight data files\n")

    phi_dict      = {}
    t_dict        = {}
    mean_phi_dict = {}
    mean_cmax_dict = {}

    for fname in data_files:
        flight_name = fname.replace("_data.py", "")
        print(f"Processing {flight_name}...")

        try:
            raw = load_data_module(os.path.join(DATA_DIR, fname))
            t_seconds, pos_xy, vel_xy, valid_mask = load_flight(raw)

            # Order parameter — works for all flights
            phi, mean_phi = compute_order_parameter(vel_xy, valid_mask)
            phi_dict[flight_name]      = phi
            t_dict[flight_name]        = t_seconds
            mean_phi_dict[flight_name] = mean_phi
            print(f"  Mean phi: {mean_phi:.3f}")

            # Mean C_max — compute for all flights regardless of threshold
            result = compute_delay_matrices(
                t_seconds=t_seconds,
                pos_xy=pos_xy,
                vel_xy=vel_xy,
                valid_mask=valid_mask,
                tau_min=-1.0,
                tau_max=1.0,
                min_points_per_tau=20,
            )
            c_vals = result.c_max[~np.isnan(result.c_max)]
            mean_cmax = float(np.mean(c_vals)) if len(c_vals) > 0 else np.nan
            mean_cmax_dict[flight_name] = mean_cmax
            print(f"  Mean C_max: {mean_cmax:.3f}")

        except Exception as e:
            print(f"  Error: {e}")

    print("\n--- Summary ---")
    for flight in sorted(mean_phi_dict):
        cmax_str = f"{mean_cmax_dict[flight]:.3f}" if flight in mean_cmax_dict else "---"
        print(f"  {flight}: phi={mean_phi_dict[flight]:.3f}  C_max={cmax_str}")

    plot_order_parameter(phi_dict, t_dict)
    plot_mean_phi_vs_mean_cmax(mean_phi_dict, mean_cmax_dict)