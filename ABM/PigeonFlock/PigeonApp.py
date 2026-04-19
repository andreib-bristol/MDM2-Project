import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import solara

from mesa.visualization import SolaraViz, Slider
from mesa.visualization.utils import update_counter

from PigeonFlock.PigeonModel import PigeonFlockModel, EMPIRICAL_TAU, HIERARCHY_ORDER

# ---------------------------------------------------------------------------
# Colour map: green (leader) → red (follower)
# ---------------------------------------------------------------------------
def _tau_to_colour(tau_i):
    t = np.clip((tau_i + 0.4) / 0.75, 0.0, 1.0)
    return (1.0 - t, t, 0.1)

BIRD_COLOURS = {bird: _tau_to_colour(tau) for bird, tau in EMPIRICAL_TAU.items()}

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
model_params = {
    "seed": {"type": "InputText", "value": 42, "label": "Random Seed"},
    "speed": Slider(label="Bird speed (m/s)", value=2.0, min=0.5, max=6.0, step=0.5),
    "perception_radius": Slider(label="Perception radius (m)", value=50.0, min=20.0, max=120.0, step=5.0),
    "separation_radius": Slider(label="Separation radius (m)", value=12.0, min=4.0, max=30.0, step=2.0),
    "w_separation": Slider(label="Separation weight", value=1.8, min=0.5, max=4.0, step=0.1),
    "w_alignment":  Slider(label="Alignment weight",  value=1.0, min=0.1, max=3.0, step=0.1),
    "w_cohesion":   Slider(label="Cohesion weight",   value=0.8, min=0.1, max=3.0, step=0.1),
    "w_homing":     Slider(label="Homing weight",     value=0.6, min=0.0, max=2.0, step=0.1),
    "start_x": Slider(label="Start X (m)", value=80.0,  min=20.0, max=380.0, step=10.0),
    "start_y": Slider(label="Start Y (m)", value=320.0, min=20.0, max=380.0, step=10.0),
    "home_x":  Slider(label="Home X (m)",  value=320.0, min=20.0, max=380.0, step=10.0),
    "home_y":  Slider(label="Home Y (m)",  value=80.0,  min=20.0, max=380.0, step=10.0),
}

# ---------------------------------------------------------------------------
# Flock visualisation
# ---------------------------------------------------------------------------
@solara.component
def FlockVisualization(model):
    update_counter.get()

    fig = Figure(figsize=(6, 6))
    ax = fig.subplots()

    W, H = model.space_width, model.space_height
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_facecolor("#e8f4f8")
    ax.set_title(f"Pigeon flock  —  step {model.steps}", fontsize=11)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # --- Centroid trail (fading) ---
    trail = model.centroid_trail
    if len(trail) > 1:
        trail_x = [p[0] for p in trail]
        trail_y = [p[1] for p in trail]
        n = len(trail)
        # Draw as segments so we can fade opacity along the trail
        for i in range(1, n):
            alpha = 0.1 + 0.7 * (i / n)   # older = more transparent
            ax.plot(
                trail_x[i-1:i+1], trail_y[i-1:i+1],
                color="steelblue", lw=1.5, alpha=alpha, solid_capstyle="round"
            )

    # --- Start marker ---
    sx, sy = model.start_pos
    ax.plot(sx, sy, marker="s", markersize=10, color="royalblue",
            markeredgecolor="white", markeredgewidth=1.5, zorder=5)
    ax.text(sx + 6, sy + 6, "Start", fontsize=8, color="royalblue")

    # --- Home marker ---
    hx, hy = model.home_pos
    ax.plot(hx, hy, marker="*", markersize=16, color="gold",
            markeredgecolor="darkorange", markeredgewidth=1.2, zorder=5)
    ax.text(hx + 6, hy + 6, "Home", fontsize=8, color="darkorange")

    # --- Leader perception circles (faint) ---
    for agent in model.agents:
        if agent.tau_i > 0.2:
            circ = Circle(agent.pos, agent.perception_radius,
                          color="green", alpha=0.04, linewidth=0)
            ax.add_patch(circ)

    # --- Birds as arrows ---
    for agent in model.agents:
        colour = BIRD_COLOURS[agent.bird_id]
        v = agent.velocity
        norm = np.linalg.norm(v)
        dx, dy = (v / norm * 6.0) if norm > 0 else (0.0, 0.0)

        ax.annotate(
            "",
            xy=(agent.pos[0] + dx, agent.pos[1] + dy),
            xytext=(agent.pos[0], agent.pos[1]),
            arrowprops=dict(arrowstyle="-|>", color=colour, lw=2),
        )
        ax.text(
            agent.pos[0] + dx * 1.4, agent.pos[1] + dy * 1.4,
            agent.bird_id, fontsize=7, ha="center", va="center",
            color=colour, fontweight="bold",
        )

    # --- Centroid + radius ---
    cx, cy = model.get_centroid()
    ax.plot(cx, cy, "k+", markersize=10, markeredgewidth=2, label="centroid")
    radius = model._flock_radius()
    circ_r = Circle((cx, cy), radius, fill=False, linestyle="--",
                     edgecolor="steelblue", linewidth=1.2, alpha=0.7,
                     label=f"radius={radius:.1f}m")
    ax.add_patch(circ_r)
    ax.legend(fontsize=8, loc="upper right")

    solara.FigureMatplotlib(fig)


# ---------------------------------------------------------------------------
# Metrics panel
# ---------------------------------------------------------------------------
@solara.component
def MetricsPanel(model):
    update_counter.get()

    fig = Figure(figsize=(6, 4))
    axes = fig.subplots(2, 1, sharex=True)
    df = model.datacollector.get_model_vars_dataframe()

    if len(df) > 1:
        steps = df.index.tolist()
        axes[0].plot(steps, df["flock_radius"].tolist(), color="steelblue", lw=1.5)
        axes[0].set_ylabel("Flock radius (m)", fontsize=9)
        axes[0].set_title("Flock cohesion over time", fontsize=10)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(steps, df["centroid_x"].tolist(), color="darkorange", lw=1.5, label="centroid x")
        axes[1].plot(steps, df["centroid_y"].tolist(), color="purple", lw=1.5, label="centroid y", linestyle="--")
        axes[1].set_ylabel("Centroid position (m)", fontsize=9)
        axes[1].set_xlabel("Step", fontsize=9)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "Run the model to see metrics",
                     ha="center", va="center", transform=axes[0].transAxes)

    fig.tight_layout()
    solara.FigureMatplotlib(fig)


# ---------------------------------------------------------------------------
# Hierarchy legend
# ---------------------------------------------------------------------------
@solara.component
def HierarchyLegend(model):
    update_counter.get()

    fig = Figure(figsize=(3, 3))
    ax = fig.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(HIERARCHY_ORDER) - 0.5)
    ax.set_title("Hierarchy (τᵢ)", fontsize=10)
    ax.axis("off")

    for rank, bird_id in enumerate(HIERARCHY_ORDER):
        tau = EMPIRICAL_TAU[bird_id]
        colour = BIRD_COLOURS[bird_id]
        y = len(HIERARCHY_ORDER) - 1 - rank
        ax.scatter([0.15], [y], s=200, color=colour, zorder=3)
        ax.text(0.28, y, f"{bird_id}  τᵢ = {tau:+.2f}s",
                va="center", fontsize=9, color="black")

    solara.FigureMatplotlib(fig)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
model = PigeonFlockModel()

page = SolaraViz(
    model,
    components=[
        (FlockVisualization, 0),
        (MetricsPanel, 1),
        (HierarchyLegend, 2),
    ],
    model_params=model_params,
    name="Pigeon Flock ABM",
)
