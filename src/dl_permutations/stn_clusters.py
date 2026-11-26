import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score, confusion_matrix,
                             normalized_mutual_info_score)

from .contrastive_trajectory_learning import (
    DATA_HASH, MODEL_PATH, N_RUNS, N_WORKERS, OUTPUT_DIR, START_CITY, config,
    drop_repeated_steps, embed, fit_encoder, load_or_generate_trajectories, make_encoder, make_tsp,
    run_single_ga, split_runs,
)

stn_config = config['stn_config']

STN_N_CENTERS = stn_config['n_centers']
STN_CENTERS_SEED = stn_config['centers_seed']

EMBED_BATCH = 4096

POPULATIONS_PATH = os.path.join(OUTPUT_DIR, f"populations_{DATA_HASH}.npz")
STN_PLOT_PATH = os.path.join(OUTPUT_DIR, "stn_clusters_k{}.png")


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


def embedding_distance(Z, centers):
    """L2 distance between unit-normalized embeddings; shape (len(Z), len(centers))."""
    return cdist(unit_norm(Z), unit_norm(centers))


def sample_centers(perms, k, seed=STN_CENTERS_SEED):
    """Indices of k distinct tours drawn from the data; tours that are the same cycle count once.

    The draw order depends only on the seed, so the centers for a smaller k are a prefix of those for a larger k.
    """
    _, first = np.unique(edge_matrix(perms), axis=0, return_index=True)
    order = np.random.default_rng(seed).permutation(np.sort(first))
    assert k <= len(order), f"only {len(order)} distinct tours, cannot draw k={k} centers"
    return order[:k]


def nearest_center(dist):
    """Nearest center per row (the lowest index wins ties) and a mask of rows whose minimum was tied.

    Breaking ties by index keeps the assignment a function of the solution, so the centers partition the space.
    """
    labels = dist.argmin(axis=1)
    ties = (dist == dist.min(axis=1, keepdims=True)).sum(axis=1) > 1
    return labels, ties


def embed_batched(model, perms, device, batch_size=EMBED_BATCH):
    return np.concatenate([embed(model, perms[i:i + batch_size], device)
                           for i in range(0, len(perms), batch_size)])


def load_or_fit_encoder(data, device):
    """Loads the contrastive encoder, training it with the existing pipeline if the checkpoint is missing."""
    if not os.path.exists(MODEL_PATH):
        return fit_encoder(data, device)
    model = make_encoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    return model


def load_inputs(device):
    """Elite trajectories without repeated steps, the encoder and unit-norm embeddings of all elites."""
    data = drop_repeated_steps(load_or_generate_trajectories())
    model = load_or_fit_encoder(data, device)
    return data, model, unit_norm(embed_batched(model, data["perm"], device))


def collapsed_sequences(labels, run_id):
    """Per-run sequences of locations (rows ordered by generation) with consecutive repeats collapsed.

    A run that stays in one location for several generations contributes a single visit and no self-loop:
    the STN records moves between locations, not time spent in one. Non-consecutive revisits are kept,
    so a run can leave a location and enter it again.
    """
    return [s[np.r_[True, s[1:] != s[:-1]]] for s in (labels[run_id == r] for r in np.unique(run_id))]


def build_stn(labels, run_id, fitness):
    """Search trajectory network over the locations given by labels, one label per representative.

    Node attributes: visits (entries of collapsed trajectories), runs (runs visiting), points (representatives
    assigned), fitness (min over visitors), start, end, best (fitness equal to the global minimum).
    Edge weight: number of transitions. Graph attributes: n_runs, best_in_strength (summed in-weight of best
    nodes) and best_strength (share of runs with at least one transition into a best node).
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

    G = nx.DiGraph(n_runs=len(seqs),
                   best_in_strength=sum(w for (_, v), w in edges.items() if best[v]),
                   best_strength=float(np.mean([best[s[1:]].any() for s in seqs])))
    for v in np.flatnonzero(points):
        G.add_node(int(v), visits=int(visits[v]), runs=int(runs[v]), points=int(points[v]),
                   fitness=float(min_fitness[v]), start=v in starts, end=v in ends, best=bool(best[v]))
    G.add_weighted_edges_from((u, v, w) for (u, v), w in edges.items())
    return G


def stn_metrics(G):
    """n_total, n_best, n_end and best_strength (Ochoa et al. 2021); n_shared is omitted, there is one algorithm."""
    return {
        "n_total": G.number_of_nodes(),
        "n_best": sum(b for _, b in G.nodes(data="best")),
        "n_end": sum(e for _, e in G.nodes(data="end")),
        "best_strength": G.graph["best_strength"],
    }


def center_layout(center_perms, seed=STN_CENTERS_SEED):
    """2-D positions of the centers from metric MDS on their pairwise edge distances."""
    D = edge_distance(center_perms, center_perms).astype(float)
    return MDS(n_components=2, metric="precomputed", init="classical_mds", random_state=seed).fit_transform(D)


def analyse_k(data, Z, k, seed=STN_CENTERS_SEED, unit=True):
    """Same k centers in both spaces: nearest-center labels, both STNs and the shared node layout. unit=False uses
    the plain L2 distance of the coordinates Z instead of that of their unit-normalized versions (the same for the
    encoders, whose embeddings have unit norm)."""
    centers = sample_centers(data["perm"], k, seed)
    perm_labels, ties = nearest_center(edge_distance(data["perm"], data["perm"][centers]))
    emb_labels, _ = nearest_center(embedding_distance(Z, Z[centers]) if unit else cdist(Z, Z[centers]))
    return {
        "k": k,
        "centers": centers,
        "perm_labels": perm_labels,
        "emb_labels": emb_labels,
        "ties": ties,
        "G_perm": build_stn(perm_labels, data["run_id"], data["fitness"]),
        "G_emb": build_stn(emb_labels, data["run_id"], data["fitness"]),
        "pos": center_layout(data["perm"][centers], seed),
    }


def label_agreement(a, b):
    """Share of identical labels, ARI, AMI and NMI between two labelings of the same points.

    NMI is not corrected for chance and grows with the number of clusters and with smaller samples; AMI is.
    """
    return {"same": float(np.mean(a == b)), "ari": adjusted_rand_score(a, b),
            "ami": adjusted_mutual_info_score(a, b), "nmi": normalized_mutual_info_score(a, b)}


def levenshtein(a, b):
    """Edit distance between two sequences (Wagner-Fischer, vectorized over the second sequence)."""
    j = np.arange(len(b) + 1)
    row = j.copy()
    for i, x in enumerate(a, 1):
        cand = np.r_[i, np.minimum(row[1:] + 1, row[:-1] + (np.asarray(b) != x))]
        row = np.minimum.accumulate(cand - j) + j
    return int(row[-1])


def sequence_agreement(seqs_a, seqs_b):
    """Per-run agreement of collapsed location sequences: mean 1 - edit distance / max length,
    share of identical sequences and share of runs ending in the same location."""
    return {
        "seq_sim": float(np.mean([1 - levenshtein(a, b) / max(len(a), len(b)) for a, b in zip(seqs_a, seqs_b)])),
        "seq_equal": float(np.mean([np.array_equal(a, b) for a, b in zip(seqs_a, seqs_b)])),
        "end_equal": float(np.mean([a[-1] == b[-1] for a, b in zip(seqs_a, seqs_b)])),
    }


def kmeans_labels(Z, k, seed=STN_CENTERS_SEED):
    """Control: k-means directly in embedding space, independent of the permutation-space centers."""
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)


def summary_row(res, data, Z, val_runs):
    """Label and trajectory agreement between the spaces plus the STN metrics of both, for one k."""
    run_id, a, b = data["run_id"], res["perm_labels"], res["emb_labels"]
    val = np.isin(run_id, val_runs)
    km = kmeans_labels(Z, res["k"])
    row = {"k": res["k"], "ties": float(res["ties"].mean()), **label_agreement(a, b),
           **{f"{m}_val": v for m, v in label_agreement(a[val], b[val]).items()},
           **sequence_agreement(collapsed_sequences(a, run_id), collapsed_sequences(b, run_id)),
           "ari_km": adjusted_rand_score(a, km), "ami_km": adjusted_mutual_info_score(a, km),
           "nmi_km": normalized_mutual_info_score(a, km)}
    for space in ("perm", "emb"):
        row.update({f"{m}_{space}": v for m, v in stn_metrics(res[f"G_{space}"]).items()})
    return row


def format_table(rows, columns):
    cells = [[f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in columns] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(columns)]
    return "\n".join(" ".join(x.rjust(w) for x, w in zip(line, widths)) for line in [columns, *cells])


def print_tables(rows):
    """Agreement between the spaces per k, then the four STN metrics in both spaces and their difference."""
    print(format_table(rows, ["k", "ties", "same", "ari", "ami", "nmi", "same_val", "ari_val", "ami_val", "nmi_val",
                              "seq_sim", "seq_equal", "end_equal", "ari_km", "ami_km", "nmi_km"]))
    print()
    metrics = [{"k": r["k"], "metric": m, "perm": r[f"{m}_perm"], "emb": r[f"{m}_emb"],
                "diff": r[f"{m}_emb"] - r[f"{m}_perm"]}
               for r in rows for m in ("n_total", "n_best", "n_end", "best_strength")]
    print(format_table(metrics, ["k", "metric", "perm", "emb", "diff"]))


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
                    linewidths=1.5, zorder=3, gid="stn_nodes")
    best = [v for v in nodes if G.nodes[v]["best"]]
    ax.scatter(*pos[best].T, marker="*", s=250, c="red", edgecolors="k", zorder=4)
    m = stn_metrics(G)
    ax.set_title(f"{title}\nn_total={m['n_total']}  n_best={m['n_best']}  n_end={m['n_end']}  "
                 f"best-strength={m['best_strength']:.2f}")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    return sc


def plot_stns(graphs, titles, pos, path=None):
    """STNs side by side with identical node positions, color scale and size scale."""
    fitness = [f for G in graphs for _, f in G.nodes(data="fitness")]
    norm = Normalize(min(fitness), max(fitness))
    max_visits = max(c for G in graphs for _, c in G.nodes(data="visits"))
    fig, axes = plt.subplots(1, len(graphs), figsize=(8 * len(graphs), 7), sharex=True, sharey=True,
                             squeeze=False)
    for ax, G, title in zip(axes[0], graphs, titles):
        sc = plot_stn(G, pos, ax, norm, max_visits, title)
    fig.colorbar(sc, ax=axes[0], label="location value (min tour length)")
    axes[0, 0].legend(handles=[
        Line2D([], [], ls="", marker="o", mfc="lightgray", mec="none", label="center"),
        Line2D([], [], ls="", marker="o", mfc="white", mec="k", label="end node"),
        Line2D([], [], ls="", marker="*", ms=12, mfc="red", mec="k", label="best node"),
    ], loc="best")
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_comparison(res, path=None):
    """Permutation-space and embedding-space STN for one k, nodes placed by MDS of the centers' edge distances."""
    return plot_stns((res["G_perm"], res["G_emb"]),
                     (f"permutation space (edge distance), k={res['k']}",
                      f"embedding space (L2), k={res['k']}"),
                     res["pos"], path)


def plot_confusion(res):
    """Row-normalized confusion matrix: permutation-space labels (rows) vs embedding-space labels (columns)."""
    k = res["k"]
    cm = confusion_matrix(res["perm_labels"], res["emb_labels"], labels=np.arange(k))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, label="share of the permutation-space location")
    ax.set_xlabel("embedding-space location")
    ax.set_ylabel("permutation-space location")
    ax.set_title(f"Location confusion, k={k}")
    return fig


def generate_populations(tsp_seed=None):
    """Re-runs the GA with the trajectory seeds and keeps every population: shape (runs, generations, pop, len);
    tsp_seed selects another TSP instance (default TSP_SEED)."""
    with ProcessPoolExecutor(N_WORKERS) as pool:
        results = list(pool.map(partial(run_single_ga, return_populations=True, tsp_seed=tsp_seed), range(N_RUNS)))
    return np.stack([populations for _, populations in results])


def load_or_generate_populations():
    if os.path.exists(POPULATIONS_PATH):
        return np.load(POPULATIONS_PATH)["populations"]
    populations = generate_populations()
    np.savez_compressed(POPULATIONS_PATH, populations=populations)
    return populations


def tour_lengths(perms):
    """TSP.total_distance vectorized over the last axis."""
    tours = full_tours(perms.reshape(-1, perms.shape[-1]))
    return make_tsp().dist_matrix[tours, np.roll(tours, -1, axis=1)].sum(axis=1).reshape(perms.shape[:-1])


def dominant_population_labels(populations, center_perms):
    """Permutation-space location holding most of each population (ties go to the lowest index) and the
    shortest tour of the population inside it; both of shape (runs, generations).

    An alternative representative to the elite: pass both, raveled, to build_stn with the run_id of the
    run-major (not repeat-dropped) trajectories.
    """
    runs, gens, size, length = populations.shape
    labels = np.stack([nearest_center(edge_distance(p.reshape(-1, length), center_perms))[0]
                       for p in populations]).reshape(runs * gens, size)
    k = len(center_perms)
    counts = np.bincount((labels + k * np.arange(runs * gens)[:, None]).ravel(), minlength=runs * gens * k)
    dominant = counts.reshape(runs * gens, k).argmax(axis=1)
    lengths = tour_lengths(populations).reshape(runs * gens, size)
    value = np.where(labels == dominant[:, None], lengths, np.inf).min(axis=1)
    return dominant.reshape(runs, gens), value.reshape(runs, gens)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, _, Z = load_inputs(device)
    val_runs, _ = split_runs()

    rows = []
    for k in STN_N_CENTERS:
        res = analyse_k(data, Z, k)
        rows.append(summary_row(res, data, Z, val_runs))
        plt.close(plot_comparison(res, STN_PLOT_PATH.format(k)))
    print_tables(rows)


if __name__ == "__main__":
    main()
