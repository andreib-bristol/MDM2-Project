import numpy as np
from correlation import compute_delay_matrices, DelayResult
from loader import load_flight


def compute_ti_ranking(result: DelayResult, c_min: float = 0.5) -> dict:
    """
    Compute per-bird flock-averaged delay t_i and rank birds into a hierarchy.

    For each bird i, t_i is the mean of tau*_ij over all j where C_max_ij >= c_min.
    A positive t_i means bird i tends to lead — its direction precedes others.
    A negative t_i means bird i tends to follow.

    This replicates the hierarchy metric from Nagy et al. (2010).

    Parameters
    ----------
    result : DelayResult
        Output from compute_delay_matrices.
    c_min : float
        Minimum peak correlation to include a pair in the t_i average.
        Nagy et al. use 0.5 as their baseline threshold.

    Returns
    -------
    dict with keys:
        'ranking' : list of (bird_id, t_i) sorted leader first
        'ti'      : dict bird_id -> t_i value (nan if no pairs passed c_min)
        'edges'   : list of (leader, follower, tau, c_max) for pairs where
                    C_max >= c_min and tau > 0 (directed leader -> follower)
    """
    N = len(result.bird_ids)
    ti = {}

    for a_idx, bird in enumerate(result.bird_ids):
        delays = []
        for b_idx in range(N):
            if a_idx == b_idx:
                continue
            c   = result.c_max[a_idx, b_idx]
            tau = result.tau_star[a_idx, b_idx]
            if np.isnan(c) or np.isnan(tau):
                continue
            if c >= c_min:
                delays.append(tau)

        ti[bird] = float(np.mean(delays)) if delays else float("nan")

    ranking = sorted(
        ti.items(),
        key=lambda x: x[1],
        reverse=True,  # most positive t_i = leader
    )

    # Directed edge i -> j exists when tau*_ij > 0 and C_max_ij >= c_min
    edges = []
    for a_idx, bird_i in enumerate(result.bird_ids):
        for b_idx, bird_j in enumerate(result.bird_ids):
            if a_idx == b_idx:
                continue
            c   = result.c_max[a_idx, b_idx]
            tau = result.tau_star[a_idx, b_idx]
            if np.isnan(c) or np.isnan(tau):
                continue
            if c >= c_min and tau > 0:
                edges.append((bird_i, bird_j, float(tau), float(c)))

    return {
        "ranking": ranking,
        "ti":      ti,
        "edges":   edges,
    }


def print_results(analysis: dict) -> None:
    """Print hierarchy ranking and directed edges to stdout."""
    print("\n--- Bird hierarchy (leader -> follower) ---")
    for rank, (bird, t) in enumerate(analysis["ranking"], 1):
        ti_str = f"{t:+.3f}" if not np.isnan(t) else "  nan"
        print(f"  {rank}. Bird {bird:>2s}   t_i = {ti_str} s")

    print("\n--- Directed edges (C_max >= threshold) ---")
    for leader, follower, tau, c in sorted(analysis["edges"], key=lambda x: -x[3]):
        print(f"  {leader} -> {follower}   tau = {tau:+.2f}s   C_max = {c:.4f}")


if __name__ == "__main__":
    
    from data.hf4_data import data

    t_seconds, pos_xy, vel_xy, valid_mask = load_flight(data)

    result = compute_delay_matrices(
        t_seconds=t_seconds,
        pos_xy=pos_xy,
        vel_xy=vel_xy,
        valid_mask=valid_mask,
        tau_min=-1.0,
        tau_max=1.0,
    )

    analysis = compute_ti_ranking(result, c_min=0.5)
    print_results(analysis)