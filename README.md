# MDM2 Group 9 — Pigeon Flock Interactions

Analysis of collective behaviour in homing pigeon flocks using GPS trajectory data from 15 flights (11 free, 4 homing).

## Dependencies

Install all required packages with:

```bash
pip install numpy pandas matplotlib scipy networkx pygraphviz
```

System dependency for network layout:
```bash
sudo apt install graphviz        # Linux
brew install graphviz            # Mac
```

## Data

Raw trajectory data is available from the original dataset repository:
https://hal.elte.hu/pigeonflocks/

Each flight folder contains one `.txt` file per bird with columns:
`t(centisec), X(m), Y(m), Z(m), dX/dt, dY/dt, dZ/dt, d²X/dt², d²Y/dt², d²Z/dt², GPS signal`

Place the downloaded `pigeonflocks_trajectories/` folder in the project root before running any scripts.

## Running the Directional Correlation Analysis

Run in order from the `main/` directory:

```bash
cd main

# Step 1 — extract and align raw data for all 15 flights
python generate_all_data.py

# Step 2 — run full cross-flight hierarchy analysis
python cross_flight.py

# Step 3 — compute Vicsek order parameter for all flights
python order_parameter.py

# Optional — visualise hierarchy network for a single flight
python visualization.py

# Optional — run hierarchy analysis for a single flight
python analysis.py
```

## File Descriptions

**`generate_all_data.py`**
Reads raw `.txt` files for all 15 flights, computes the overlapping time window across all birds in each flight, and saves aligned data as `{flight}_data.py` files in `main/data/`.

**`loader.py`**
Converts raw flight data into numpy arrays. Applies the windowed GPS quality filter from Nagy et al. (2010): a timepoint is included only if at least 2 of the 5 surrounding GPS fixes are real measurements. Also truncates birds to a common array length and warns when GPS quality is poor.

**`correlation.py`**
Computes pairwise directional correlation matrices for a single flight. For each bird pair (i, j), sweeps a lag grid τ ∈ [-1.0, +1.0]s and finds the delay τ*_ij at which C_ij(τ) is maximised. Returns τ* and C_max matrices. Filters by GPS quality, speed, and inter-bird distance.

**`analysis.py`**
Takes the output of `correlation.py` and computes per-bird flock-averaged delays t_i. Ranks birds by t_i to produce the hierarchy and builds directed leader-follower edges where C_max ≥ 0.5 and τ* > 0.

**`visualization.py`**
Plots the leader-follower network using NetworkX and Graphviz dot layout. Node colour encodes t_i (green = leader, red = follower). Edge width scales with C_max. Edge labels show τ* delay in seconds.

**`cross_flight.py`**
Runs the full pipeline across all 15 flights. Excludes flights with mean C_max < 0.5. Computes pairwise Spearman rank correlations of t_i values between flights with bootstrap confidence intervals (1000 resamples). Runs a paired t-test comparing leadership scores between free and homing flights. Produces a heatmap of t_i values across all passing flights.

**`order_parameter.py`**
Computes the Vicsek order parameter φ(t) = |Σ u_i(t)| / N for all 15 flights. Plots φ(t) time series and a scatter plot of mean φ vs mean C_max across all flights to validate both coordination measures independently.

## Key Results

- 6 of 15 flights passed the coordination quality threshold (mean C_max ≥ 0.5): all 4 homing flights and ff8, ff11
- Flight hf4 produced a clear 9-bird hierarchy: A > H > B > D > J > L > C > I > G
- Cross-flight Spearman correlation hf3 vs hf4: r_s = +0.671, 95% CI [+0.103, +0.975]
- Paired t-test: no significant difference in leadership scores between free and homing flights (t = 0.220, p = 0.832)
- Order parameter and directional correlation agree strongly across all 15 flights (Pearson r = +0.960, p < 0.001)

## Reference

Nagy, M., Ákos, Zs., Biro, D., & Vicsek, T. (2010). Hierarchical group dynamics in pigeon flocks. Nature, 464, 890-893.