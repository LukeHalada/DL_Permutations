"""Search trajectory networks (STN) with clustered locations.

k centers are drawn from the elites; every tour belongs to its nearest center, in the permutation space by edge
distance and in a representation (embedding or coordinates) by L2 distance. Each space gets its own STN over the same
centers, so the two can be compared node by node.
"""
import os
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
from sklearn.manifold import MDS

from .contrastive_trajectory_learning import START_CITY

with open(os.path.join(os.path.dirname(__file__), "stn_config.yaml")) as f:
    stn_config = yaml.safe_load(f)

CENTERS_SEED = stn_config['stn_config']['centers_seed']


def full_tours(perms):
    """Prepends the fixed start cities (TSP.start), giving complete cycles of shape (m, n)."""
    perms = np.atleast_2d(perms)
    return np.hstack([np.tile(np.arange(START_CITY), (len(perms), 1)), perms])


def edge_matrix(perms):
    """Binary undirected edge incidence matrix of full tours: one row per tour, n(n-1)/2 columns."""
    tours = full_tours(perms)
    m, n = tours.shape
    nxt = np.roll(tours, -1, axis=1)
    lo, hi = np.minimum(tours, nxt), np.maximum(tours, nxt)
    X = np.zeros((m, n * (n - 1) // 2), dtype=np.float32)
    np.put_along_axis(X, lo * n - lo * (lo + 1) // 2 + hi - lo - 1, 1, axis=1)
    return X


def edge_distance(a, b):
    """Edge distance between full tours, n minus the number of shared undirected edges; shape (len(a), len(b))."""
    n = START_CITY + np.shape(a)[-1]
    return (n - edge_matrix(a) @ edge_matrix(b).T).astype(int)


def unit_norm(Z):
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def sample_centers(perms, k, seed=CENTERS_SEED):
    """Indices of k distinct tours drawn from the data; tours that are the same cycle count once. The draw order
    depends only on the seed, so the centers for a smaller k are a prefix of those for a larger k."""
    _, first = np.unique(edge_matrix(perms), axis=0, return_index=True)
    order = np.random.default_rng(seed).permutation(np.sort(first))
    assert k <= len(order), f"only {len(order)} distinct tours, cannot draw k={k} centers"
    return order[:k]


def collapsed_sequences(labels, run_id):
    """Per run, the sequence of locations with consecutive repeats removed."""
    seqs = []
    for run in np.unique(run_id):
        seq = labels[run_id == run]
        seqs.append(seq[np.r_[True, seq[1:] != seq[:-1]]])
    return seqs


def build_stn(labels, run_id, fitness):
    """Search trajectory network over the locations given by labels, one label per elite.

    Node attributes: visits (entries of collapsed trajectories), runs (runs visiting), points (elites assigned),
    fitness (min over visitors), start, end, best (fitness equal to the global minimum). Edge weight: number of
    transitions. Graph attribute best_strength: share of runs with at least one transition into a best node.
    """
    seqs = collapsed_sequences(labels, run_id)
    size = labels.max() + 1
    min_fitness = np.full(size, np.inf)
    np.minimum.at(min_fitness, labels, fitness)
    visits = np.bincount(np.concatenate(seqs), minlength=size)
    runs = np.bincount(np.unique(np.c_[run_id, labels], axis=0)[:, 1], minlength=size)
    points = np.bincount(labels, minlength=size)
    best = np.isclose(min_fitness, fitness.min())
    starts, ends = {s[0] for s in seqs}, {s[-1] for s in seqs}
    edges = Counter((int(u), int(v)) for s in seqs for u, v in zip(s[:-1], s[1:]))

    G = nx.DiGraph(best_strength=float(np.mean([best[s[1:]].any() for s in seqs])))
    for v in np.flatnonzero(points):
        G.add_node(int(v), visits=int(visits[v]), runs=int(runs[v]), points=int(points[v]),
                   fitness=float(min_fitness[v]), start=v in starts, end=v in ends, best=bool(best[v]))
    G.add_weighted_edges_from((u, v, w) for (u, v), w in edges.items())
    return G


def center_layout(center_perms, seed=CENTERS_SEED):
    """2-D positions of the centers from metric MDS on their pairwise edge distances."""
    D = edge_distance(center_perms, center_perms).astype(float)
    return MDS(n_components=2, metric="precomputed", init="classical_mds", random_state=seed).fit_transform(D)


def analyse_k(data, Z, k):
    """Same k centers in the permutation space and in Z (L2): nearest-center labels, both STNs, the node layout."""
    centers = sample_centers(data["perm"], k)
    perm_labels = edge_distance(data["perm"], data["perm"][centers]).argmin(axis=1)
    emb_labels = cdist(Z, Z[centers]).argmin(axis=1)
    return {
        "k": k,
        "centers": centers,
        "perm_labels": perm_labels,
        "emb_labels": emb_labels,
        "G_perm": build_stn(perm_labels, data["run_id"], data["fitness"]),
        "G_emb": build_stn(emb_labels, data["run_id"], data["fitness"]),
        "pos": center_layout(data["perm"][centers]),
    }


def plot_stn(G, pos, ax, norm, max_visits, title):
    """Draws an STN at fixed positions: size = visits, color = min fitness, black rim = end node, star = best."""
    nodes = list(G)
    sizes = 30 + 600 * np.array([G.nodes[v]["visits"] for v in nodes]) / max_visits
    radius = dict(zip(nodes, np.sqrt(sizes) / 2))
    ax.scatter(*pos.T, s=8, c="lightgray", zorder=1)
    for u, v, w in G.edges(data="weight"):
        ax.annotate("", pos[v], pos[u], zorder=2,
                    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.6, lw=0.4 + 0.8 * np.log1p(w),
                                    shrinkA=radius[u], shrinkB=radius[v], connectionstyle="arc3,rad=0.15"))
    sc = ax.scatter(*pos[nodes].T, s=sizes, c=[G.nodes[v]["fitness"] for v in nodes], cmap="viridis_r",
                    norm=norm, edgecolors=["k" if G.nodes[v]["end"] else "none" for v in nodes],
                    linewidths=1.5, zorder=3)
    best = [v for v in nodes if G.nodes[v]["best"]]
    ax.scatter(*pos[best].T, marker="*", s=250, c="red", edgecolors="k", zorder=4)
    n_end = sum(e for _, e in G.nodes(data="end"))
    ax.set_title(f"{title}\nn_end={n_end}  best-strength={G.graph['best_strength']:.2f}")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    return sc


def plot_stns(graphs, titles, pos):
    """STNs side by side with identical node positions, color scale and size scale."""
    fitness = [f for G in graphs for _, f in G.nodes(data="fitness")]
    norm = Normalize(min(fitness), max(fitness))
    max_visits = max(c for G in graphs for _, c in G.nodes(data="visits"))
    fig, axes = plt.subplots(1, len(graphs), figsize=(8 * len(graphs), 7), sharex=True, sharey=True, squeeze=False)
    for ax, G, title in zip(axes[0], graphs, titles):
        sc = plot_stn(G, pos, ax, norm, max_visits, title)
    fig.colorbar(sc, ax=axes[0], label="location value (min tour length)")
    axes[0, 0].legend(handles=[
        Line2D([], [], ls="", marker="o", mfc="lightgray", mec="none", label="center"),
        Line2D([], [], ls="", marker="o", mfc="white", mec="k", label="end node"),
        Line2D([], [], ls="", marker="*", ms=12, mfc="red", mec="k", label="best node"),
    ], loc="best")
    return fig
