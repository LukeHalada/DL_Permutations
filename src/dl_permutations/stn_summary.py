"""STN summary: the STN of the permutation space against the STNs of encoder embeddings and of compressions without
learning, on held-out GA runs. Metrics, baselines, replication over TSP instances and figures of
notebooks/stn_summary.ipynb.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import stn_encoders as E
from .stn_clusters import analyse_k, edge_distance, edge_matrix, plot_stns, stn_config

METRIC_KS = stn_config['stn_config']['metric_ks']
FIGURE_KS = stn_config['stn_config']['figure_ks']
BASELINE_DIM = stn_config['stn_config']['baseline_dim']
METRICS = ("ARI", "soft", "spearman", "recall@5", "funnel", "end_equal", "gen_r2")
REFERENCE = f"classical MDS({BASELINE_DIM})"


# --- metrics

def per_tour_soft(data, Z, centers):
    """Per tour, the Spearman correlation of its distances to the centers in both spaces (nan if all equal)."""
    a = rankdata(edge_distance(data["perm"], data["perm"][centers]), axis=1)
    b = rankdata(cdist(Z, Z[centers]), axis=1)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))


def recall_at(Z, perms, k=5):
    """Share of the k nearest distinct tours in Z that lie within the k-th smallest edge distance of the tour."""
    idx = np.sort(np.unique(edge_matrix(perms), axis=0, return_index=True)[1])
    D = edge_distance(perms[idx], perms[idx]).astype(float)
    L = cdist(Z[idx], Z[idx])
    np.fill_diagonal(D, np.inf)
    np.fill_diagonal(L, np.inf)
    nearest = np.argsort(L, axis=1)[:, :k]
    return float((np.take_along_axis(D, nearest, axis=1) <= np.sort(D, axis=1)[:, k - 1:k]).mean())


def generation_r2(data, Z, folds=5):
    """Out-of-fold R^2 of a ridge regression of the generation on Z (standardized; folds grouped by run)."""
    y = data["generation"].astype(float)
    pred = np.zeros(len(y))
    for train, test in GroupKFold(folds).split(Z, groups=data["run_id"]):
        pred[test] = make_pipeline(StandardScaler(), Ridge()).fit(Z[train], y[train]).predict(Z[test])
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def metrics(data, Z):
    """Agreement of the coordinates Z (L2) with the permutation space on a set of GA runs; the k-dependent ones are
    averaged over METRIC_KS.

    ARI: of the locations (nearest of the k centers) in both spaces. soft: per-tour Spearman correlation of the
    distances to the k centers, averaged over tours. spearman: Spearman correlation of all pair distances.
    recall@5: see recall_at. funnel: -|best-strength of the STN of Z - best-strength of the permutation-space STN|.
    end_equal: share of runs whose last elite lands in the same location in both spaces. gen_r2: generation probe.
    """
    last = np.r_[data["run_id"][1:] != data["run_id"][:-1], True]
    per_k = []
    for k in METRIC_KS:
        res = analyse_k(data, Z, k)
        per_k.append([adjusted_rand_score(res["perm_labels"], res["emb_labels"]),
                      np.nanmean(per_tour_soft(data, Z, res["centers"])),
                      -abs(res["G_emb"].graph["best_strength"] - res["G_perm"].graph["best_strength"]),
                      np.mean(res["perm_labels"][last] == res["emb_labels"][last])])
    ari, soft, funnel, end_equal = np.mean(per_k, axis=0)
    i, j = np.triu_indices(len(Z), 1)
    spearman = spearmanr(cdist(Z, Z)[i, j], edge_distance(data["perm"], data["perm"])[i, j])[0]
    return {"ARI": ari, "soft": soft, "spearman": float(spearman), "recall@5": recall_at(Z, data["perm"]),
            "funnel": funnel, "end_equal": end_equal, "gen_r2": generation_r2(data, Z)}


def summary_rows(results):
    """results: name -> list of metric dicts (one per seed). One row per name: the mean and, for several seeds, the
    sd of every metric."""
    rows = []
    for name, runs in results.items():
        x = np.array([[r[m] for m in METRICS] for r in runs])
        row = {"variant": name, "n": len(x), **dict(zip(METRICS, x.mean(axis=0)))}
        if len(x) > 1:
            row.update({f"{m} sd": v for m, v in zip(METRICS, x.std(axis=0, ddof=1))})
        rows.append(row)
    return rows


def format_rows(rows):
    """Text table of summary_rows: 'mean' or 'mean ± sd' per metric."""
    columns = ["variant", *METRICS]
    cells = [[str(r["variant"])] + [f"{r[m]:.3f}" + (f" ± {r[m + ' sd']:.3f}" if m + " sd" in r else "")
                                    for m in METRICS] for r in rows]
    widths = [max(len(c), *(len(line[i]) for line in cells)) for i, c in enumerate(columns)]
    return "\n".join("  ".join(v.rjust(w) for v, w in zip(line, widths)) for line in [columns] + cells)


# --- compressions without learning

class ClassicalMDS:
    """Classical (Torgerson) MDS of the edge distances of the fitting tours to the top dim positive eigenvalues; new
    tours are placed with Gower's out-of-sample formula (exact for the fitting tours)."""

    def __init__(self, dim):
        self.dim = dim

    def fit(self, perms):
        D2 = edge_distance(perms, perms).astype(float) ** 2
        self.col_mean = D2.mean(axis=0)
        B = -0.5 * (D2 - self.col_mean[None, :] - D2.mean(axis=1)[:, None] + D2.mean())
        vals, vecs = np.linalg.eigh(B)
        order = np.argsort(vals)[::-1][:self.dim]
        keep = order[vals[order] > 0]
        self.vals, self.vecs, self.perms = vals[keep], vecs[:, keep], perms
        return self

    def transform(self, perms):
        d2 = edge_distance(perms, self.perms).astype(float) ** 2
        return 0.5 * (self.col_mean[None, :] - d2) @ self.vecs / np.sqrt(self.vals)


def baseline_coordinates(perms, fit_perms, dim=BASELINE_DIM):
    """Coordinates of the tours perms without learning, fitted on fit_perms: classical MDS of the edge distances and
    PCA of the edge incidence vectors (whose squared L2 distance is twice the edge distance)."""
    X_fit, X = edge_matrix(fit_perms), edge_matrix(perms)
    return {REFERENCE: ClassicalMDS(dim).fit(fit_perms).transform(perms),
            f"PCA({dim}) of incidence": PCA(dim, svd_solver="full").fit(X_fit).transform(X),
            "PCA(2) of incidence": PCA(2, svd_solver="full").fit(X_fit).transform(X)}


# --- replication

def replication(device, arch="incidence_mlp"):
    """Per replication instance, on its test runs: the metrics of the reference, of PCA(64) and of the encoder trained
    on the instance (seed 0). Returns instance -> name -> metric dict."""
    out = {}
    for instance in list(E.INSTANCES)[1:]:
        data, fit = E.test_runs(instance), E.training_elites(instance)
        coords = baseline_coordinates(data["perm"], fit["perm"])
        coords = {name: coords[name] for name in (REFERENCE, f"PCA({BASELINE_DIM}) of incidence")}
        coords[arch] = E.embed_tours(E.load_or_train(arch, device, instance), data["perm"], device)
        out[instance] = {name: metrics(data, Z) for name, Z in coords.items()}
    return out


# --- figures

def stn_figure(data, coords, k, title):
    """STN of the permutation space next to the STNs of the given coordinates (name -> Z) for one k: the same
    centers, node positions (metric MDS of the centers' edge distances), color and size scales."""
    results = [analyse_k(data, Z, k) for Z in coords.values()]
    fig = plot_stns([results[0]["G_perm"]] + [r["G_emb"] for r in results],
                    [f"permutation space (edge distance), k={k}"] + [f"{name}, k={k}" for name in coords],
                    results[0]["pos"])
    fig.suptitle(title, y=1.03)
    return fig


def trajectories_figure(data, coords_fit, coords, runs, title):
    """Trajectories of some runs in the first two principal components of each representation (name -> (fitting
    coordinates, coordinates of data)), PCA fitted on the fitting coordinates; points colored by generation."""
    norm = Normalize(0, data["generation"].max())
    fig, axes = plt.subplots(1, len(coords), figsize=(8.5 * len(coords), 7), squeeze=False)
    for ax, name in zip(axes[0], coords):
        pca = PCA(2, svd_solver="full").fit(coords_fit[name])
        xy = pca.transform(coords[name])
        ax.scatter(*xy.T, s=4, c="lightgray", zorder=1, label="every elite of the set")
        for run in runs:
            m = np.flatnonzero(data["run_id"] == run)
            line, = ax.plot(*xy[m].T, lw=0.9, alpha=0.8, zorder=2, label=f"run {run}")
            sc = ax.scatter(*xy[m].T, c=data["generation"][m], cmap="viridis", norm=norm, s=22,
                            edgecolors=line.get_color(), linewidths=0.8, zorder=3)
            ax.scatter(*xy[m[0]], marker="*", s=220, color=line.get_color(), edgecolors="k", zorder=4)
            ax.scatter(*xy[m[-1]], marker="s", s=70, color=line.get_color(), edgecolors="k", zorder=4)
        ratio = pca.explained_variance_ratio_
        ax.set(title=name, xlabel=f"PC 1 ({ratio[0]:.0%} of the variance)", ylabel=f"PC 2 ({ratio[1]:.0%})")
    handles, _ = axes[0, 0].get_legend_handles_labels()
    handles += [Line2D([], [], ls="", marker="*", ms=12, mfc="white", mec="k", label="first elite"),
                Line2D([], [], ls="", marker="s", ms=7, mfc="white", mec="k", label="last elite")]
    axes[0, 0].legend(handles=handles, loc="best", fontsize=8)
    fig.colorbar(sc, ax=axes[0], label="generation")
    fig.suptitle(title, y=0.98)
    return fig
