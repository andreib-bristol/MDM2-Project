from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


BirdID = str


@dataclass(frozen=True)
class DelayResult:
    """Stores pairwise delay and correlation results for a single flight."""
    bird_ids: List[BirdID]
    tau_grid: np.ndarray       # (L,) seconds
    tau_star: np.ndarray       # (N,N) best lag in seconds, nan on diagonal / insufficient data
    c_max: np.ndarray          # (N,N) peak correlation, nan on diagonal / insufficient data
    n_used_at_peak: np.ndarray # (N,N) number of timepoints used at the peak lag


def generate_test_data() -> Tuple[np.ndarray, dict, dict, dict]:
    """
    Generate synthetic flock data for testing.
    - Bird A leads with smooth turning motion
    - Bird B follows A with a 0.4s delay
    - Bird C is mostly independent noise
    """
    dt = 0.2
    T = 300
    t_seconds = np.arange(0, T * dt, dt)

    angle_A = 0.5 * np.sin(0.5 * t_seconds) + 0.3 * np.sin(0.1 * t_seconds)
    vel_A = np.column_stack((np.cos(angle_A), np.sin(angle_A)))

    delay_steps = 2
    angle_B = np.roll(angle_A, delay_steps)
    vel_B = np.column_stack((np.cos(angle_B), np.sin(angle_B)))

    angle_C = 0.8 * np.sin(0.2 * t_seconds + 1.0)
    vel_C = np.column_stack((np.cos(angle_C), np.sin(angle_C)))

    vel_A += 0.05 * np.random.randn(*vel_A.shape)
    vel_B += 0.05 * np.random.randn(*vel_B.shape)
    vel_C += 0.20 * np.random.randn(*vel_C.shape)

    pos_A = np.cumsum(vel_A * dt, axis=0)
    pos_B = np.cumsum(vel_B * dt, axis=0)
    pos_C = np.cumsum(vel_C * dt, axis=0)

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

    Parameters
    ----------
    v_xy : (T, 2) velocity array
    eps  : speed threshold below which a vector is considered stationary

    Returns
    -------
    u_xy           : (T, 2) unit vectors
    good_speed_mask: (T,) bool, True where speed > eps
    """
    speed = np.linalg.norm(v_xy, axis=1)
    good = speed > eps
    u = np.zeros_like(v_xy)
    u[good] = v_xy[good] / speed[good, None]
    return u, good


def _corr_for_lag(
    bird_i: BirdID,
    bird_j: BirdID,
    tau_idx: int,
    lag_steps: np.ndarray,
    u: dict,
    valid_mask: dict,
    good_speed: dict,
    pos_xy: dict,
    distance_limit_m: float,
    min_points_per_tau: int,
) -> Tuple[float, int]:
    """
    Compute directional correlation C_ij at a single lag index.

    C_ij(tau) = mean( u_i(t) . u_j(t + tau) ) over valid, close timepoints.

    A positive tau means bird_i's direction preceded bird_j's — bird_i leads.

    Parameters
    ----------
    bird_i, bird_j  : bird identifiers
    tau_idx         : index into lag_steps array
    lag_steps       : array of integer lag steps corresponding to tau grid
    u               : dict of unit velocity arrays per bird
    valid_mask      : dict of GPS validity masks per bird
    good_speed      : dict of speed validity masks per bird
    pos_xy          : dict of position arrays per bird
    distance_limit_m: maximum allowed distance between birds (metres)
    min_points_per_tau: minimum valid samples required to return a value

    Returns
    -------
    (correlation, n_used) — correlation is nan if fewer than min_points_per_tau
    """
    lag = lag_steps[tau_idx]

    if lag == 0:
        ui = u[bird_i]
        uj = u[bird_j]
        valid = (
            valid_mask[bird_i] & valid_mask[bird_j] &
            good_speed[bird_i] & good_speed[bird_j]
        )
        dij = np.linalg.norm(pos_xy[bird_i] - pos_xy[bird_j], axis=1)

    elif lag > 0:
        # bird_i at time t, bird_j at time t + lag
        ui = u[bird_i][:-lag]
        uj = u[bird_j][lag:]
        valid = (
            valid_mask[bird_i][:-lag] & valid_mask[bird_j][lag:] &
            good_speed[bird_i][:-lag] & good_speed[bird_j][lag:]
        )
        dij = np.linalg.norm(pos_xy[bird_i][:-lag] - pos_xy[bird_j][:-lag], axis=1)

    else:
        kk = -lag
        # bird_i at time t, bird_j at time t - kk
        ui = u[bird_i][kk:]
        uj = u[bird_j][:-kk]
        valid = (
            valid_mask[bird_i][kk:] & valid_mask[bird_j][:-kk] &
            good_speed[bird_i][kk:] & good_speed[bird_j][:-kk]
        )
        dij = np.linalg.norm(pos_xy[bird_i][kk:] - pos_xy[bird_j][kk:], axis=1)

    valid &= (dij < distance_limit_m)
    n = int(np.count_nonzero(valid))

    if n < min_points_per_tau:
        return np.nan, n

    dots = np.einsum("ij,ij->i", ui, uj)
    return float(np.mean(dots[valid])), n


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
    Compute pairwise directional correlation delay matrices for a single flight.

    For each pair of birds (i, j), sweeps a lag grid and finds the delay
    tau*_ij at which C_ij(tau) = mean(u_i(t) . u_j(t+tau)) is maximised.
    A positive tau*_ij means bird i's direction preceded bird j's — i leads j.

    Parameters
    ----------
    t_seconds        : (T,) time array in seconds, assumed roughly uniform
    pos_xy, vel_xy   : dict bird_id -> (T, 2) position and velocity arrays
    valid_mask       : dict bird_id -> (T,) bool, True where GPS fix is real
    tau_min, tau_max : lag search range in seconds
    tau_step         : lag step in seconds. Defaults to data timestep
    distance_limit_m : exclude timepoints where birds are further apart than this
    speed_eps        : exclude timepoints where speed is below this threshold
    min_points_per_tau: minimum valid samples to compute C_ij at a given lag

    Returns
    -------
    DelayResult with tau_star, c_max, and n_used_at_peak matrices
    """
    bird_ids = sorted(pos_xy.keys())
    if set(bird_ids) != set(vel_xy.keys()) or set(bird_ids) != set(valid_mask.keys()):
        raise ValueError("pos_xy, vel_xy, valid_mask must have the same bird IDs")

    T = t_seconds.shape[0]
    dt_data = float(np.median(np.diff(t_seconds)))
    if tau_step is None:
        tau_step = dt_data

    tau_grid = np.arange(tau_min, tau_max + 0.5 * tau_step, tau_step, dtype=float)
    lag_steps = np.rint(tau_grid / dt_data).astype(int)

    N = len(bird_ids)
    tau_star     = np.full((N, N), np.nan, dtype=float)
    c_max        = np.full((N, N), np.nan, dtype=float)
    n_used_at_peak = np.zeros((N, N), dtype=int)

    u = {}
    good_speed = {}
    for b in bird_ids:
        if vel_xy[b].shape != (T, 2) or pos_xy[b].shape != (T, 2) or valid_mask[b].shape != (T,):
            raise ValueError(f"Bad shape for bird {b}. Expected pos/vel (T,2), valid (T,)")
        u[b], good_speed[b] = unit_vectors(vel_xy[b], eps=speed_eps)

    for a_idx, bird_i in enumerate(bird_ids):
        for b_idx, bird_j in enumerate(bird_ids):
            if bird_i == bird_j:
                continue

            C      = np.full(tau_grid.shape, np.nan, dtype=float)
            n_used = np.zeros(tau_grid.shape, dtype=int)

            for tau_idx in range(len(tau_grid)):
                C[tau_idx], n_used[tau_idx] = _corr_for_lag(
                    bird_i, bird_j, tau_idx, lag_steps,
                    u, valid_mask, good_speed, pos_xy,
                    distance_limit_m, min_points_per_tau,
                )

            if np.all(np.isnan(C)):
                continue

            k_star = int(np.nanargmax(C))
            c_max[a_idx, b_idx]          = C[k_star]
            tau_star[a_idx, b_idx]       = tau_grid[k_star]
            n_used_at_peak[a_idx, b_idx] = int(n_used[k_star])

    return DelayResult(
        bird_ids=bird_ids,
        tau_grid=tau_grid,
        tau_star=tau_star,
        c_max=c_max,
        n_used_at_peak=n_used_at_peak,
    )


if __name__ == "__main__":
    t_seconds, pos_xy, vel_xy, valid_mask = generate_test_data()

    res = compute_delay_matrices(
        t_seconds=t_seconds,
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