import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx

from correlation import compute_delay_matrices
from loader import load_flight
from analysis import compute_ti_ranking


def build_network(analysis: dict) -> nx.DiGraph:
    """
    Build a directed NetworkX graph from the hierarchy analysis.

    Nodes represent birds, weighted by t_i.
    Edges represent directed leader -> follower relationships
    where C_max >= c_min and tau* > 0.

    Parameters
    ----------
    analysis : dict
        Output from compute_ti_ranking.

    Returns
    -------
    nx.DiGraph with node attribute 'ti' and edge attributes 'tau', 'c_max'
    """
    G = nx.DiGraph()

    for bird, ti in analysis["ti"].items():
        G.add_node(bird, ti=ti)

    for leader, follower, tau, c_max in analysis["edges"]:
        G.add_edge(leader, follower, tau=tau, c_max=c_max)

    return G


def plot_hierarchy_network(analysis: dict, flight_name: str = "hf4",
                           output_path: str = "hierarchy_network.png") -> None:
    """
    Plot the leader-follower hierarchy as a directed network graph.

    Node colour encodes t_i (green = leader, red = follower).
    Edge width scales with C_max strength.
    Edge labels show tau* delay in seconds.
    Layout uses graphviz dot algorithm for top-to-bottom hierarchy,
    with a fallback to a manual t_i-based layout if graphviz is unavailable.

    Parameters
    ----------
    analysis    : dict from compute_ti_ranking
    flight_name : used in the plot title
    output_path : filepath to save the figure
    """
    G = build_network(analysis)
    ranking = analysis["ranking"]

    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog="dot")
    except ImportError:
        print("Warning: graphviz not found, using fallback layout. "
              "Install with: sudo apt install graphviz && pip install pygraphviz")
        ranking_order = [bird for bird, _ in ranking]
        pos = {
            bird: (i * 2.0, -analysis["ti"][bird] * 10)
            for i, bird in enumerate(ranking_order)
        }

    ti_values = np.array([analysis["ti"][b] for b, _ in ranking])
    norm = mcolors.Normalize(ti_values.min(), ti_values.max())
    cmap = plt.colormaps["RdYlGn"]

    node_colors  = [cmap(norm(analysis["ti"][bird])) for bird in G.nodes()]
    edge_weights = [G[u][v]["c_max"] for u, v in G.edges()]
    edge_taus    = [G[u][v]["tau"]   for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(12, 8))

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=2400,
        alpha=0.9,
    )

    nx.draw_networkx_labels(G, pos, ax=ax, font_size=16, font_weight="bold")

    

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=[w * 3 for w in edge_weights],
        alpha=0.6,
        edge_color=edge_taus,
        edge_cmap=plt.colormaps["Blues"],
        arrows=True,
        arrowsize=20,
        connectionstyle="arc3,rad=0.1",
    )

    

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    

    ax.set_title(
        f"Leader-follower network ({flight_name})",
        fontsize=24,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {output_path}")


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
    plot_hierarchy_network(analysis, flight_name="hf4")