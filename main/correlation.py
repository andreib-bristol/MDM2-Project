# correlation.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional



BirdID = str


@dataclass(frozen=True)
class DelayResult:
    bird_ids: List[BirdID]
    tau_grid: np.ndarray          # (L,) seconds
    tau_star: np.ndarray          # (N,N) best lag (seconds), nan on diag / insufficient data
    c_max: np.ndarray             # (N,N) peak correlation, nan on diag / insufficient data
    n_used_at_peak: np.ndarray    # (N,N) number of timepoints used at the peak lag

def generate_test_data():
    """
    Generate synthetic flock data:
    - A leads
    - B follows A with 0.4 s delay
    - C is mostly noise
    """

    dt = 0.2
    T = 300  # 60 seconds
    t_seconds = np.arange(0, T*dt, dt)

    # Leader (A) makes smooth turning motion
    angle_A = 0.5 * np.sin(0.5 * t_seconds) + 0.3 * np.sin(0.1 * t_seconds)
    vel_A = np.column_stack((np.cos(angle_A), np.sin(angle_A)))

    # B follows A with 0.4 s delay (2 timesteps)
    delay_steps = 2
    angle_B = np.roll(angle_A, delay_steps)
    vel_B = np.column_stack((np.cos(angle_B), np.sin(angle_B)))

    # C = mostly random motion
    angle_C = 0.8 * np.sin(0.2 * t_seconds + 1.0)
    vel_C = np.column_stack((np.cos(angle_C), np.sin(angle_C)))

    # Add small noise
    vel_A += 0.05 * np.random.randn(*vel_A.shape)
    vel_B += 0.05 * np.random.randn(*vel_B.shape)
    vel_C += 0.2 * np.random.randn(*vel_C.shape)

    # Fake positions by integrating velocity
    pos_A = np.cumsum(vel_A * dt, axis=0)
    pos_B = np.cumsum(vel_B * dt, axis=0)
    pos_C = np.cumsum(vel_C * dt, axis=0)

    # All birds always valid (for now)
    valid_mask = {
        "A": np.ones(T, dtype=bool),
        "B": np.ones(T, dtype=bool),
        "C": np.ones(T, dtype=bool),
    }

    pos_xy = {"A": pos_A, "B": pos_B, "C": pos_C}
    vel_xy = {"A": vel_A, "B": vel_B, "C": vel_C}

    return t_seconds, pos_xy, vel_xy, valid_mask

def unit_vectors(v_xy: np.ndarray, eps: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert velocity vectors to unit direction vectors.
    Returns (u_xy, good_speed_mask).
      u_xy: (T,2)
      good_speed_mask: (T,) bool where speed > eps
    """
    speed = np.linalg.norm(v_xy, axis=1)
    good = speed > eps
    u = np.zeros_like(v_xy)
    u[good] = v_xy[good] / speed[good, None]
    return u, good


def _shift_for_lag(arr: np.ndarray, lag_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align arrays for dot-product at a given lag.
    We want dot(u_i[t], u_j[t + lag_steps]) over overlapping indices.

    Returns (a, b) with matching length along axis=0.
    """
    if lag_steps == 0:
        return arr, arr
    if lag_steps > 0:
        # compare arr[t] with arr[t+lag]
        a = arr[:-lag_steps]
        b = arr[lag_steps:]
        return a, b
    else:
        k = -lag_steps
        # compare arr[t] with arr[t-k]  <=> arr[t+k] with arr[t]
        a = arr[k:]
        b = arr[:-k]
        return a, b





def compute_delay_matrices(
    t_seconds: np.ndarray,
    pos_xy: Dict[BirdID, np.ndarray],
    vel_xy: Dict[BirdID, np.ndarray],
    valid_mask: Dict[BirdID, np.ndarray],
    *,
    tau_min: float = -2.0,
    tau_max: float = 2.0,
    tau_step: Optional[float] = None,
    distance_limit_m: float = 100.0,
    speed_eps: float = 1e-6,
    min_points_per_tau: int = 50,
) -> DelayResult:
    """
    Compute tau_star and C_max matrices for a single flight.

    Parameters
    ----------
    t_seconds:
        (T,) time array, assumed roughly uniform.
    pos_xy, vel_xy:
        dict bird_id -> (T,2)
    valid_mask:
        dict bird_id -> (T,) bool, computed by data-processing person
    tau_min, tau_max:
        lag range in seconds
    tau_step:
        lag step in seconds. If None, uses data dt (median diff of t_seconds)
    distance_limit_m:
        apply d_ij(t) < distance_limit_m at the "reference" time index used for dot products
    speed_eps:
        additional filter: ignore times where speed is ~0 when forming unit vectors
    min_points_per_tau:
        minimum number of usable samples required to compute C_ij(tau). Else returns nan for that tau.

    Returns
    -------
    DelayResult
    """
    bird_ids = sorted(pos_xy.keys())
    if set(bird_ids) != set(vel_xy.keys()) or set(bird_ids) != set(valid_mask.keys()):
        raise ValueError("pos_xy, vel_xy, valid_mask must have the same bird IDs")

    T = t_seconds.shape[0]
    dt_data = float(np.median(np.diff(t_seconds)))
    if tau_step is None:
        tau_step = dt_data

    # lag grid (seconds) and integer lag steps
    tau_grid = np.arange(tau_min, tau_max + 0.5 * tau_step, tau_step, dtype=float)
    lag_steps = np.rint(tau_grid / dt_data).astype(int)  # map tau to nearest integer step

    N = len(bird_ids)
    tau_star = np.full((N, N), np.nan, dtype=float)
    c_max = np.full((N, N), np.nan, dtype=float)
    n_used_at_peak = np.zeros((N, N), dtype=int)

    # Precompute unit velocity directions and "good speed" masks
    u = {}
    good_speed = {}
    for b in bird_ids:
        if vel_xy[b].shape != (T, 2) or pos_xy[b].shape != (T, 2) or valid_mask[b].shape != (T,):
            raise ValueError(f"Bad shape for bird {b}. Expect pos/vel (T,2), valid (T,)")
        u[b], good_speed[b] = unit_vectors(vel_xy[b], eps=speed_eps)

    # Helper: compute correlation for a given pair and lag
    def corr_for_lag(i: BirdID, j: BirdID, k: int) -> Tuple[float, int]:
        """
        Return (C_ij(tau_k), n_used) at lag step k.
        We compute dot(u_i[t], u_j[t+lag]) over valid & close times.
        """
        lag = lag_steps[k]
        # Align arrays for overlapping indices
        ui_a, ui_b = _shift_for_lag(u[i], lag)   # NOTE: returns arr, arr but we need two different arrays
        # We'll do shifting explicitly for i and j:
        if lag == 0:
            ui = u[i]
            uj = u[j]
            valid = (
                valid_mask[i] & valid_mask[j] &
                good_speed[i] & good_speed[j]
            )
            # distance at time t
            dij = np.linalg.norm(pos_xy[i] - pos_xy[j], axis=1)
            valid &= (dij < distance_limit_m)
        elif lag > 0:
            # compare i[t] with j[t+lag]
            ui = u[i][:-lag]
            uj = u[j][lag:]
            valid = (
                valid_mask[i][:-lag] & valid_mask[j][lag:] &
                good_speed[i][:-lag] & good_speed[j][lag:]
            )
            dij = np.linalg.norm(pos_xy[i][:-lag] - pos_xy[j][:-lag], axis=1)  # distance at time t
            valid &= (dij < distance_limit_m)
        else:
            kk = -lag
            # compare i[t] with j[t-kk]  (equivalently i[t+kk] with j[t])
            ui = u[i][kk:]
            uj = u[j][:-kk]
            valid = (
                valid_mask[i][kk:] & valid_mask[j][:-kk] &
                good_speed[i][kk:] & good_speed[j][:-kk]
            )
            dij = np.linalg.norm(pos_xy[i][kk:] - pos_xy[j][kk:], axis=1)  # distance at time t (aligned with ui index)
            valid &= (dij < distance_limit_m)

        n = int(np.count_nonzero(valid))
        if n < min_points_per_tau:
            return (np.nan, n)

        dots = np.einsum("ij,ij->i", ui, uj)  # dot per time
        return (float(np.mean(dots[valid])), n)

    # Main loop over pairs
    for a_idx, i in enumerate(bird_ids):
        for b_idx, j in enumerate(bird_ids):
            if i == j:
                continue

            C = np.full(tau_grid.shape, np.nan, dtype=float)
            n_used = np.zeros(tau_grid.shape, dtype=int)

            for k in range(len(tau_grid)):
                C[k], n_used[k] = corr_for_lag(i, j, k)

            # pick peak
            if np.all(np.isnan(C)):
                continue
            k_star = int(np.nanargmax(C))
            c_max[a_idx, b_idx] = C[k_star]
            tau_star[a_idx, b_idx] = tau_grid[k_star]
            n_used_at_peak[a_idx, b_idx] = int(n_used[k_star])

    return DelayResult(
        bird_ids=bird_ids,
        tau_grid=tau_grid,
        tau_star=tau_star,
        c_max=c_max,
        n_used_at_peak=n_used_at_peak,
    )

rt, pos_xy, vel_xy, valid_mask = generate_test_data()

res = compute_delay_matrices(
    t_seconds=rt,
    pos_xy=pos_xy,
    vel_xy=vel_xy,
    valid_mask=valid_mask,
    tau_min=-1.0,
    tau_max=1.0,
)

print("Bird order:", res.bird_ids)
print("Tau* matrix (seconds):")
print(res.tau_star)
print("C_max matrix:")
print(res.c_max)