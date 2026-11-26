"""Final iteration: selection on the dev set with a run bootstrap, replication over instances, and the project table.

Criterion of a trained encoder on a set of runs (its gain over its own random init, same architecture and seed):
the mean of four gains, each averaged over SELECTION_KS where it depends on k,
- soft agreement (per-tour Spearman of the distances to the k centers, permutation space vs embedding). It is partly
  the training target itself: every objective here orders triplets by edge distance, so a higher soft agreement is
  expected from training and is not independent evidence;
- funnel: -|best_strength of the embedding STN - best_strength of the permutation-space STN| (closer is better);
- end_equal: share of runs whose final elite lands in the same location in both spaces;
- generation probe: out-of-fold R^2 of a ridge regression of the generation on the embedding (folds grouped by run,
  R^2 pooled over the out-of-fold predictions), clipped at 0: a probe worse than the mean carries no linear
  information, and the unbounded negative values of untrained encoders would otherwise dominate the mean.

All four are computed from per-run pieces (centers, labels and probe predictions fixed on the full set), so the runs
can be resampled: the bootstrap draws the runs of the set with replacement and recomputes every criterion on the same
draws. Selection rule: the variant with the highest mean criterion over its seeds wins unless another is tied with it,
i.e. the difference of the means is at most se_factor bootstrap standard errors of that difference; among tied variants
the simplest wins. The variants, their order of simplicity, the k values and the replication instances are fixed in
the config and below before any result of this iteration was computed.

Added after the results, at the reviewer's request: a run-bootstrap test of the differences in the absolute summary
metrics (ARI, soft agreement, recall@5, Spearman) between two variants (absolute_difference). The criterion is a gain
over the own random init and rewards a weak start, so the choice of an architecture needs the absolute metrics.
Also added then: the final figures of the default variant (stn_pair_figure, trajectories_pca_figure), and the
baselines without learning (PCA of the edge incidence vectors to 64 and 2 dimensions, classical MDS of the edge
distances to 64 dimensions; fitted on the elites of the training runs, evaluated with the plain L2 distance).
Then, at the reviewer's decision, classical MDS(64) became the reference (upper bound without learning):
baseline_replication over the replication instances, stn_three_panel_figure.
"""
import hashlib
import os
import platform
import sys
from importlib import metadata

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .contrastive_trajectory_learning import OUTPUT_DIR, config, drop_repeated_steps, split_runs
from .encoder_architectures import (
    ARCHITECTURES, PROTOCOL_SEEDS, REPLICATION, embed, instance_elites, load_or_fit, load_or_generate_dev_runs,
    load_or_generate_test_runs, own_init, random_init,
)
from .encoder_variants import (
    V1_DIR, distance_correlation, distinct_cycles, embed_tours, load_encoder, local_metrics, runs_subset,
)
from .stn_clusters import analyse_k, edge_distance, edge_matrix, plot_comparison, plot_stns
from .stn_controls import location_entries

selection_config = config['selection_config']
SELECTION_KS = selection_config['ks']
BOOTSTRAP = selection_config['bootstrap']
BOOTSTRAP_SEED = selection_config['bootstrap_seed']
SE_FACTOR = selection_config['se_factor']
PROBE_FOLDS = 5
COMPONENTS = ("soft", "funnel", "end_equal", "gen_r2")
SUMMARY_METRICS = ("ARI k>=20", "soft k>=20", "recall@5", "spearman")

# Every variant of instance 1 in the final iteration: row label -> (arch, objective, local weight). Row labels start
# with the architecture, trained rows are '<label> s<seed>', their random inits 'random init <arch> s<seed>'.
FINAL_VARIANTS = {
    "lstm": ("lstm", "edge", 0.0),
    "incidence_mlp": ("incidence_mlp", "edge", 0.0),
    "edge_deepsets": ("edge_deepsets", "edge", 0.0),
    "incidence_mlp w=1": ("incidence_mlp", "edge", 1.0),
    "incidence_mlp populations": ("incidence_mlp", "population_edge", 0.0),
    "incidence_mlp populations + local (dropped)": ("incidence_mlp", "population_edge", 1.0),
}
# Selections made by the final rule on the dev set: candidate -> FINAL_VARIANTS label, simplest first. The default of
# the project is incidence_mlp, edge objective, w = 0 (decided after the previous iteration); these selections check
# it on the dev set. populations + local term is dropped and only reported.
FINAL_SELECTIONS = {
    "architecture (edge objective, w = 0)": {"incidence_mlp": "incidence_mlp", "edge_deepsets": "edge_deepsets",
                                             "lstm": "lstm"},
    "local term (incidence_mlp, edge objective)": {"w=0": "incidence_mlp", "w=1": "incidence_mlp w=1"},
    "training data (incidence_mlp, w = 0)": {"elites": "incidence_mlp", "populations": "incidence_mlp populations"},
}
REPLICATION_INSTANCES = [f"tsp{seed}" for seed in REPLICATION['tsp_seeds']]
# final figures of the default variant (added after the results)
FINAL_FIGURE_PATH = os.path.join(OUTPUT_DIR, "final_{}.png")


def final_sets():
    """Evaluation sets of instance 1 in the final iteration: validation runs (20), dev runs (selection), test runs."""
    val_runs, _ = split_runs()
    return {"validation": runs_subset(instance_elites("tsp21"), val_runs),
            "dev": drop_repeated_steps(load_or_generate_dev_runs()),
            "test": drop_repeated_steps(load_or_generate_test_runs())}


def final_models(device, variants=None, seeds=PROTOCOL_SEEDS):
    """Every final variant of instance 1 for every seed, with the random inits of all architectures."""
    variants = FINAL_VARIANTS if variants is None else variants
    models = {f"random init {arch} s{seed}": random_init(arch, device, seed) for arch in ARCHITECTURES for seed in seeds}
    for label, (arch, objective, weight) in variants.items():
        for seed in seeds:
            models[f"{label} s{seed}"] = load_or_fit(arch, device, objective=objective, seed=seed, local_weight=weight)
    return models


def gen_oof(Z, data, folds=PROBE_FOLDS):
    """Out-of-fold ridge predictions of the generation (standardized features, folds grouped by run)."""
    pred = np.zeros(len(Z))
    n_folds = min(folds, len(np.unique(data["run_id"])))
    for train, test in GroupKFold(n_folds).split(Z, groups=data["run_id"]):
        model = make_pipeline(StandardScaler(), Ridge()).fit(Z[train], data["generation"][train])
        pred[test] = model.predict(Z[test])
    return pred


def per_tour_soft(data, Z, centers):
    """Per-tour Spearman correlation of the distances to the centers (nan when a tour's distances are all equal)."""
    a = rankdata(edge_distance(data["perm"], data["perm"][centers]), axis=1)
    b = rankdata(cdist(Z, Z[centers]), axis=1)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))


def entered_best(labels, data):
    """Per run (np.unique order): whether it enters a location holding the best tour (as best_strength of build_stn)."""
    entered, share = location_entries(labels, data["run_id"])
    min_fitness = np.full(len(share), np.inf)
    np.minimum.at(min_fitness, labels, data["fitness"])
    return entered[:, np.isclose(min_fitness, data["fitness"].min())].any(axis=1).astype(float)


def per_run_pieces(data, Z, ks=SELECTION_KS, unit=True):
    """Per-run pieces of the criterion of one embedding on one set, runs in np.unique order.

    soft_sum / soft_count (runs, k): sum and number of the per-tour soft agreements of each run; end_equal, best_emb,
    best_perm (runs, k): whether the run's final elite lands in the same location in both spaces and whether the run
    enters a best location in the embedding / permutation-space STN; gen_pred / gen_true / row_run: out-of-fold
    predictions of the generation per row and the run of each row; ari (k,): ARI of the two partitions. unit=False:
    the embedding labels use the plain L2 distance of Z (the baselines), not that of the unit-normalized Z.
    """
    runs, run_idx = np.unique(data["run_id"], return_inverse=True)
    last = np.r_[data["run_id"][1:] != data["run_id"][:-1], True]
    soft_sum, soft_count, end_equal, best_emb, best_perm, ari = [], [], [], [], [], []
    for k in ks:
        res = analyse_k(data, Z, k, unit=unit)
        r = per_tour_soft(data, Z, res["centers"])
        soft_sum.append(np.bincount(run_idx, np.nan_to_num(r), minlength=len(runs)))
        soft_count.append(np.bincount(run_idx, ~np.isnan(r), minlength=len(runs)))
        same = np.zeros(len(runs))
        same[run_idx[last]] = res["perm_labels"][last] == res["emb_labels"][last]
        end_equal.append(same)
        best_emb.append(entered_best(res["emb_labels"], data))
        best_perm.append(entered_best(res["perm_labels"], data))
        ari.append(adjusted_rand_score(res["perm_labels"], res["emb_labels"]))
    return {"soft_sum": np.stack(soft_sum, 1), "soft_count": np.stack(soft_count, 1), "end_equal": np.stack(end_equal, 1),
            "best_emb": np.stack(best_emb, 1), "best_perm": np.stack(best_perm, 1), "gen_pred": gen_oof(Z, data),
            "gen_true": data["generation"].astype(float), "row_run": run_idx, "ari": np.array(ari)}


def components(pieces, W):
    """The four criterion components (before taking gains) for run weights W (B, runs); returns (B, 4)."""
    total = W.sum(axis=1)[:, None]
    soft = (W @ pieces["soft_sum"]) / (W @ pieces["soft_count"])
    end_equal = (W @ pieces["end_equal"]) / total
    funnel = -np.abs(W @ pieces["best_emb"] - W @ pieces["best_perm"]) / total
    w_rows = W[:, pieces["row_run"]]
    y, p = pieces["gen_true"], pieces["gen_pred"]
    mean_y = (w_rows @ y) / w_rows.sum(axis=1)
    sse = w_rows @ (y - p) ** 2
    sst = (w_rows * (y[None, :] - mean_y[:, None]) ** 2).sum(axis=1)
    return np.stack([soft.mean(axis=1), funnel.mean(axis=1), end_equal.mean(axis=1), np.maximum(1 - sse / sst, 0)],
                    axis=1)


def bootstrap_weights(n_runs, b=BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Row 0: every run once (the full set); rows 1..b: multiplicities of resamples of the runs with replacement."""
    rng = np.random.default_rng(seed)
    return np.vstack([np.ones(n_runs), rng.multinomial(n_runs, np.full(n_runs, 1 / n_runs), size=b)])


def gains(pieces, name, W):
    """Gains of a trained row over its own random init, per weight row: (B, 4)."""
    return components(pieces[name], W) - components(pieces[own_init(name)], W)


def evaluate_set(models, data, device, ks=SELECTION_KS, noncanonical=()):
    """Per-run pieces and summary metrics (ARI and soft agreement averaged over ks, recall@5, Spearman of all pair
    distances) of every model; rows in noncanonical are embedded without reorienting the tours."""
    pieces, summary = {}, {}
    for name, model in models.items():
        Z = embed_tours(model, data["perm"], device, canonical=False) if name in noncanonical else \
            embed(model, data["perm"], device)
        p = pieces[name] = per_run_pieces(data, Z, ks)
        summary[name] = {"ARI k>=20": float(p["ari"].mean()),
                         "soft k>=20": float(np.mean(p["soft_sum"].sum(0) / p["soft_count"].sum(0))),
                         "recall@5": local_metrics(Z, data["perm"], ks=(5,))["recall@5"],
                         "spearman": distance_correlation(Z, data["perm"])}
    return pieces, summary


def seed_rows(label, pieces):
    return sorted(n for n in pieces if n.startswith(label + " s") and n[len(label) + 2:].isdigit())


def select(pieces, candidates, W, se_factor=SE_FACTOR):
    """The final selection rule on one set. candidates: candidate -> FINAL_VARIANTS label, simplest first. Returns the
    choice and one row per candidate: mean criterion over seeds (full set), seed sd, its bootstrap SE, the difference
    to the top with the bootstrap SE of that difference, whether it is tied with the top, and the mean gain per
    component."""
    score, seed_sd, comp = {}, {}, {}
    for cand, label in candidates.items():
        g = np.stack([gains(pieces, name, W) for name in seed_rows(label, pieces)])   # (seeds, B, 4)
        score[cand] = g.mean(axis=2).mean(axis=0)
        seed_sd[cand] = float(g[:, 0].mean(axis=1).std(ddof=1))
        comp[cand] = g[:, 0].mean(axis=0)
    top = max(candidates, key=lambda cand: score[cand][0])
    rows = []
    for cand in candidates:
        diff = score[top] - score[cand]
        se = float(diff[1:].std(ddof=1))
        rows.append({"variant": cand, "criterion": float(score[cand][0]), "seed sd": seed_sd[cand],
                     "boot SE": float(score[cand][1:].std(ddof=1)), "diff to top": float(diff[0]), "SE of diff": se,
                     "tied with top": bool(cand == top or diff[0] <= se_factor * se),
                     **{f"Δ{c}": float(comp[cand][i]) for i, c in enumerate(COMPONENTS)}})
    choice = next(r["variant"] for r in rows if r["tied with top"])
    return choice, rows


def paired_difference(pieces, label_a, label_b, W, component=None):
    """Mean difference a - b over seeds on the full set of the criterion (or of one component index), its bootstrap SE
    (runs resampled) and the seed sd of the per-seed differences (seeds paired by number)."""
    pick = (lambda g: g.mean(axis=1)) if component is None else (lambda g: g[:, component])
    a = np.stack([pick(gains(pieces, n, W)) for n in seed_rows(label_a, pieces)])
    b = np.stack([pick(gains(pieces, n, W)) for n in seed_rows(label_b, pieces)])
    diff = a - b
    return {"diff": float(diff[:, 0].mean()), "boot SE": float(diff[:, 1:].mean(axis=0).std(ddof=1)),
            "seed sd": float(diff[:, 0].std(ddof=1)) if len(diff) > 1 else float("nan")}


def table_rows(pieces, summary, W, labels, stage, set_name):
    """Project-table rows of trained variants on one set: the criterion and its components as gains over the own
    random init (mean over seeds, seed sd, bootstrap SE over runs) and the absolute summary metrics."""
    rows = []
    for label in labels:
        names = seed_rows(label, pieces)
        g = np.stack([gains(pieces, n, W) for n in names])   # (seeds, B, 4)
        crit = g.mean(axis=2)
        rows.append({"stage": stage, "set": set_name, "variant": label, "n": len(names),
                     "criterion": float(crit[:, 0].mean()),
                     "seed sd": float(crit[:, 0].std(ddof=1)) if len(names) > 1 else float("nan"),
                     "boot SE": float(crit[:, 1:].mean(axis=0).std(ddof=1)),
                     **{f"Δ{c}": float(g[:, 0, i].mean()) for i, c in enumerate(COMPONENTS)},
                     **{m: float(np.mean([summary[n][m] for n in names])) for m in SUMMARY_METRICS}})
    return rows


def init_rows(summary, stage, set_name, seeds=PROTOCOL_SEEDS):
    """Absolute summary metrics of the random inits (mean over seeds) for the project table."""
    rows = []
    for arch in ARCHITECTURES:
        names = [f"random init {arch} s{seed}" for seed in seeds if f"random init {arch} s{seed}" in summary]
        if names:
            rows.append({"stage": stage, "set": set_name, "variant": f"random init {arch}", "n": len(names),
                         **{m: float(np.mean([summary[n][m] for n in names])) for m in SUMMARY_METRICS}})
    return rows


# Added after the results of the final iteration, at the reviewer's request (not part of the pre-registered rule):
# a run-bootstrap test of the differences in the absolute summary metrics between two variants.

def per_cycle_recall(Z, perms, k=5):
    """recall@k of local_metrics per distinct cycle: the first-occurrence indices of the cycles and their values (the
    mean of the values is local_metrics' recall@k)."""
    idx = distinct_cycles(perms)
    D = edge_distance(perms[idx], perms[idx]).astype(float)
    E = cdist(Z[idx], Z[idx])
    np.fill_diagonal(D, np.inf)
    np.fill_diagonal(E, np.inf)
    nearest = np.argsort(E, axis=1)
    return idx, (np.take_along_axis(D, nearest[:, :k], axis=1) <= np.sort(D, axis=1)[:, k - 1:k]).mean(axis=1)


def ari_from_contingency(C):
    """Adjusted Rand index from contingency tables (..., k, k), as sklearn's adjusted_rand_score of the rows."""
    def pairs(x):
        return x * (x - 1) / 2
    index = pairs(C).sum(axis=(-2, -1))
    a = pairs(C.sum(axis=-1)).sum(axis=-1)
    b = pairs(C.sum(axis=-2)).sum(axis=-1)
    expected = a * b / pairs(C.sum(axis=(-2, -1)))
    return (index - expected) / ((a + b) / 2 - expected)


def absolute_pieces(data, Z, ks=SELECTION_KS, recall_k=5, unit=True):
    """Per-run pieces of the absolute summary metrics (SUMMARY_METRICS) of one embedding on one set, every structure
    fixed on the full set as for the criterion: per k the centers and both labelings (ARI from the per-run contingency
    tables) and the per-tour soft agreements; recall@5 per distinct cycle (neighbours among the distinct cycles of the
    full set, a cycle belongs to the run of its first occurrence); the ranks of all pair distances (Spearman = Pearson
    correlation of the ranks, a pair weighted by the product of the multiplicities of the runs of its two tours).
    unit as in per_run_pieces."""
    runs, run_idx = np.unique(data["run_id"], return_inverse=True)
    R = len(runs)
    contingency, soft_sum, soft_count = [], [], []
    for k in ks:
        res = analyse_k(data, Z, k, unit=unit)
        C = np.zeros((R, k, k))
        np.add.at(C, (run_idx, res["perm_labels"], res["emb_labels"]), 1)
        contingency.append(C)
        r = per_tour_soft(data, Z, res["centers"])
        soft_sum.append(np.bincount(run_idx, np.nan_to_num(r), minlength=R))
        soft_count.append(np.bincount(run_idx, ~np.isnan(r), minlength=R))
    idx, recall = per_cycle_recall(Z, data["perm"], recall_k)
    i, j = np.triu_indices(len(Z), 1)
    x = rankdata(cdist(Z, Z)[i, j])
    y = rankdata(edge_distance(data["perm"], data["perm"])[i, j])
    x -= x.mean()
    y -= y.mean()
    block = np.minimum(run_idx[i], run_idx[j]) * R + np.maximum(run_idx[i], run_idx[j])
    pair_stats = np.stack([np.bincount(block, w, minlength=R * R) for w in (np.ones_like(x), x, y, x * x, y * y, x * y)],
                          axis=1)
    return {"contingency": contingency, "soft_sum": np.stack(soft_sum, 1), "soft_count": np.stack(soft_count, 1),
            "recall_sum": np.bincount(run_idx[idx], recall, minlength=R),
            "recall_count": np.bincount(run_idx[idx], minlength=R).astype(float), "pair_stats": pair_stats}


def absolute_metrics(p, W):
    """The SUMMARY_METRICS of one embedding for run weights W (B, runs): (B, 4); a row of ones is the full set."""
    ari = np.mean([ari_from_contingency(np.einsum("br,rij->bij", W, C)) for C in p["contingency"]], axis=0)
    soft = ((W @ p["soft_sum"]) / (W @ p["soft_count"])).mean(axis=1)
    recall = (W @ p["recall_sum"]) / (W @ p["recall_count"])
    n, sx, sy, sxx, syy, sxy = ((W[:, :, None] * W[:, None, :]).reshape(len(W), -1) @ p["pair_stats"]).T
    spearman = (sxy / n - sx * sy / n ** 2) / np.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2))
    return np.stack([ari, soft, recall, spearman], axis=1)


def absolute_difference(abs_pieces, label_a, label_b, W, set_name=""):
    """Per summary metric: the means of a and b over seeds (full set), the difference a - b (seeds paired by number),
    its bootstrap SE and 95% percentile interval (runs resampled, the same draws for every seed and both variants),
    the seed sd of the per-seed differences and the verdict of the list (solid when |difference| exceeds twice the
    larger of the seed sd and the bootstrap SE)."""
    a = np.stack([absolute_metrics(abs_pieces[n], W) for n in seed_rows(label_a, abs_pieces)])   # (seeds, B, 4)
    b = np.stack([absolute_metrics(abs_pieces[n], W) for n in seed_rows(label_b, abs_pieces)])
    diff = a - b
    boot = diff[:, 1:].mean(axis=0)
    rows = []
    for m, metric in enumerate(SUMMARY_METRICS):
        d, se, sd = float(diff[:, 0, m].mean()), float(boot[:, m].std(ddof=1)), float(diff[:, 0, m].std(ddof=1))
        low, high = np.percentile(boot[:, m], [2.5, 97.5])
        rows.append({"set": set_name, "metric": metric, label_a: float(a[:, 0, m].mean()),
                     label_b: float(b[:, 0, m].mean()), "difference": d, "boot SE": se, "95% low": float(low),
                     "95% high": float(high), "seed sd": sd, "verdict": verdict(d, max(sd, se))})
    return rows



# Final figures of the default variant, added after the results at the reviewer's request.

def stn_pair_figure(data, Z, k, title):
    """STN of the permutation space next to the STN of the embedding for one k: the same k centers, the same node
    positions (metric MDS of the centers' edge distances), the same color and size scales (plot_comparison)."""
    fig = plot_comparison(analyse_k(data, Z, k))
    fig.suptitle(title, y=1.03)
    return fig


def trajectories_pca_figure(data, Z, runs, title):
    """Trajectories of some runs in the first two principal components of the embedding (PCA fitted on every elite
    of the set, drawn in gray): a line per run, its elites colored by generation on one scale, star = first elite,
    square = last elite."""
    pca = PCA(n_components=2).fit(Z)
    xy = pca.transform(Z)
    norm = Normalize(0, data["generation"].max())
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(*xy.T, s=4, c="lightgray", zorder=1, label="every elite of the set")
    for run in runs:
        m = np.flatnonzero(data["run_id"] == run)
        line, = ax.plot(*xy[m].T, lw=0.9, alpha=0.8, zorder=2, label=f"run {run}")
        sc = ax.scatter(*xy[m].T, c=data["generation"][m], cmap="viridis", norm=norm, s=22,
                        edgecolors=line.get_color(), linewidths=0.8, zorder=3)
        ax.scatter(*xy[m[0]], marker="*", s=220, color=line.get_color(), edgecolors="k", zorder=4)
        ax.scatter(*xy[m[-1]], marker="s", s=70, color=line.get_color(), edgecolors="k", zorder=4)
    fig.colorbar(sc, ax=ax, label="generation")
    ratio = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC 1 ({ratio[0]:.0%} of the variance)")
    ax.set_ylabel(f"PC 2 ({ratio[1]:.0%} of the variance)")
    ax.set_title(title)
    handles, _ = ax.get_legend_handles_labels()
    handles += [Line2D([], [], ls="", marker="*", ms=12, mfc="white", mec="k", label="first elite"),
                Line2D([], [], ls="", marker="s", ms=7, mfc="white", mec="k", label="last elite")]
    ax.legend(handles=handles, loc="best", fontsize=8)
    return fig



def trajectories_pair_figure(data, coords, ratios, titles, runs):
    """Trajectories of some runs in two 2-D projections side by side (one generation color scale): per panel the
    coordinates of every elite of the set (gray) and the explained variance ratios of its two axes."""
    norm = Normalize(0, data["generation"].max())
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for ax, xy, ratio, title in zip(axes, coords, ratios, titles):
        ax.scatter(*xy.T, s=4, c="lightgray", zorder=1, label="every elite of the set")
        for run in runs:
            m = np.flatnonzero(data["run_id"] == run)
            line, = ax.plot(*xy[m].T, lw=0.9, alpha=0.8, zorder=2, label=f"run {run}")
            sc = ax.scatter(*xy[m].T, c=data["generation"][m], cmap="viridis", norm=norm, s=22,
                            edgecolors=line.get_color(), linewidths=0.8, zorder=3)
            ax.scatter(*xy[m[0]], marker="*", s=220, color=line.get_color(), edgecolors="k", zorder=4)
            ax.scatter(*xy[m[-1]], marker="s", s=70, color=line.get_color(), edgecolors="k", zorder=4)
        ax.set_xlabel(f"PC 1 ({ratio[0]:.0%} of the variance)")
        ax.set_ylabel(f"PC 2 ({ratio[1]:.0%} of the variance)")
        ax.set_title(title)
    handles, _ = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], ls="", marker="*", ms=12, mfc="white", mec="k", label="first elite"),
                Line2D([], [], ls="", marker="s", ms=7, mfc="white", mec="k", label="last elite")]
    axes[0].legend(handles=handles, loc="best", fontsize=8)
    fig.colorbar(sc, ax=axes, label="generation")
    return fig


# Baselines without learning, added after the results at the reviewer's request: compressions of the tours fitted
# on the elites of the training runs of instance 1 (the runs every encoder was trained on), evaluated on the same
# sets, centers and metrics as the encoders, with the plain L2 distance of their coordinates.
BASELINE_DIM = config['model_config']['out_dim']
ABSOLUTE_COMPONENTS = ("funnel", "end_equal", "gen_r2")
BASELINE_METRICS = ("ARI k>=20", "soft k>=20", "spearman", "recall@5", "funnel", "end_equal", "gen_r2")


def baseline_fit_elites():
    """Elites of the training runs of instance 1 (split_runs): the data of the encoders' training triplets."""
    _, train_runs = split_runs()
    return runs_subset(instance_elites("tsp21"), train_runs)


def pca_2d(fit, X):
    """The first two principal components of X (PCA fitted on fit) and their explained variance ratios."""
    pca = PCA(2, svd_solver="full").fit(fit)
    return pca.transform(X), pca.explained_variance_ratio_


class ClassicalMDS:
    """Classical (Torgerson) MDS of the edge distances of the fitting tours, to the top dim positive eigenvalues;
    new tours are placed with Gower's out-of-sample formula, exact for the fitting tours. The edge distance is not
    Euclidean (negative eigenvalues), negative_mass reports their share of the positive mass."""

    def __init__(self, dim):
        self.dim = dim

    def fit(self, perms):
        D2 = edge_distance(perms, perms).astype(float) ** 2
        self.col_mean = D2.mean(axis=0)
        B = -0.5 * (D2 - self.col_mean[None, :] - D2.mean(axis=1)[:, None] + D2.mean())
        vals, vecs = np.linalg.eigh(B)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        keep = np.flatnonzero(vals[:self.dim] > 0)
        self.vals, self.vecs, self.perms = vals[keep], vecs[:, keep], perms
        self.kept_share = float(self.vals.sum() / vals[vals > 0].sum())
        self.negative_mass = float(-vals[vals < 0].sum() / vals[vals > 0].sum())
        return self

    def transform(self, perms):
        d2 = edge_distance(perms, self.perms).astype(float) ** 2
        return 0.5 * (self.col_mean[None, :] - d2) @ self.vecs / np.sqrt(self.vals)


def baseline_embeddings(perms, fit_perms, dim=BASELINE_DIM):
    """Coordinates of the tours perms under the baselines fitted on fit_perms, and the classical MDS fit."""
    X_fit, X = edge_matrix(fit_perms), edge_matrix(perms)
    mds = ClassicalMDS(dim).fit(fit_perms)
    return {f"PCA({dim}) of incidence": PCA(dim, svd_solver="full").fit(X_fit).transform(X),
            "PCA(2) of incidence": pca_2d(X_fit, X)[0],
            f"classical MDS({dim}) of edge distance": mds.transform(perms)}, mds


def metric_matrix(data, Z, W, unit=True):
    """BASELINE_METRICS of coordinates Z on a set for run weights W: (B, 7); the funnel, end_equal and the
    generation probe are absolute values here (not gains)."""
    a = absolute_metrics(absolute_pieces(data, Z, unit=unit), W)        # ARI, soft, recall@5, Spearman
    c = components(per_run_pieces(data, Z, unit=unit), W)              # soft, funnel, end_equal, gen_r2
    return np.column_stack([a[:, 0], a[:, 1], a[:, 3], a[:, 2], c[:, 1], c[:, 2], c[:, 3]])


def baseline_comparison(data, reference, baselines, W, set_name, reference_label="incidence_mlp"):
    """reference: seed row name -> embedding of the trained reference; baselines: name -> coordinates (plain L2).
    Returns the value rows (reference: mean and seed sd over seeds; baselines: value on the full set) and, per
    baseline and metric, the difference baseline - reference (paired per seed, then averaged), its bootstrap SE and
    95% percentile interval (runs resampled), the seed sd and the verdict: 'beats' / 'below' the reference when the
    difference exceeds twice the larger of seed sd and bootstrap SE, else 'matches (within noise)'."""
    ref = np.stack([metric_matrix(data, Z, W) for Z in reference.values()])   # (seeds, B, 7)
    def sd(x):
        return float(x.std(ddof=1)) if len(x) > 1 else float("nan")
    n = len(reference)
    values = [{"set": set_name, "variant": f"{reference_label} ({n} seed{'s' if n > 1 else ''})",
               **{m: float(ref[:, 0, i].mean()) for i, m in enumerate(BASELINE_METRICS)},
               **{f"{m} seed sd": sd(ref[:, 0, i]) for i, m in enumerate(BASELINE_METRICS)}}]
    diffs = []
    for name, Z in baselines.items():
        b = metric_matrix(data, Z, W, unit=False)
        values.append({"set": set_name, "variant": name, **{m: float(b[0, i]) for i, m in enumerate(BASELINE_METRICS)}})
        diff = b[None] - ref
        boot = diff[:, 1:].mean(axis=0)
        for i, m in enumerate(BASELINE_METRICS):
            d, se, seed_sd = float(diff[:, 0, i].mean()), float(boot[:, i].std(ddof=1)), sd(diff[:, 0, i])
            low, high = np.percentile(boot[:, i], [2.5, 97.5])
            noise = se if np.isnan(seed_sd) else max(se, seed_sd)
            diffs.append({"set": set_name, "baseline": name, "metric": m, "difference": d, "boot SE": se,
                          "95% low": float(low), "95% high": float(high), "seed sd": seed_sd,
                          "vs reference": "beats" if d > 2 * noise else "below" if d < -2 * noise else
                          "matches (within noise)"})
    return values, diffs


def absolute_component_values(pieces, names):
    """Absolute funnel, end_equal and generation probe on the full set, mean over the given rows of pieces."""
    v = np.mean([components(pieces[n], np.ones((1, len(pieces[n]["end_equal"]))))[0, 1:] for n in names], axis=0)
    return dict(zip(ABSOLUTE_COMPONENTS, map(float, v)))



# Added after the baselines, at the reviewer's request: classical MDS(64) is the reference of the project (upper
# bound without learning), replicated with PCA(64) over the replication instances.
REFERENCE_BASELINE = f"classical MDS({BASELINE_DIM}) of edge distance"
REPLICATION_BASELINES = (REFERENCE_BASELINE, f"PCA({BASELINE_DIM}) of incidence")
AGREEMENT_METRICS = BASELINE_METRICS[:6]   # agreement with the permutation space; gen_r2 is a probe


def baseline_replication(device, instances=REPLICATION_INSTANCES, baselines=REPLICATION_BASELINES):
    """Per replication instance, on its test runs: BASELINE_METRICS of incidence_mlp trained on the elites of the
    instance (replication seed, nothing trained here) and of the baselines fitted on the elites of the instance's
    training runs (plain L2), and per baseline and metric the difference baseline - incidence_mlp with its bootstrap
    SE over the runs (baseline_comparison; 'set' is the instance)."""
    arch, seed = REPLICATION['arch'], REPLICATION['seed']
    _, train_runs = split_runs()
    values, diffs = [], []
    for instance in instances:
        data = drop_repeated_steps(load_or_generate_test_runs(instance))
        fit = runs_subset(instance_elites(instance), train_runs)
        model = load_or_fit(arch, device, objective="edge", instance=instance, seed=seed)
        coords, _ = baseline_embeddings(data["perm"], fit["perm"])
        v, d = baseline_comparison(data, {f"{arch} s{seed}": embed(model, data["perm"], device)},
                                   {name: coords[name] for name in baselines},
                                   bootstrap_weights(len(np.unique(data["run_id"]))), instance)
        values += v
        diffs += d
    return values, diffs


def over_instances(values, diffs):
    """Mean and sd over instances of every metric per variant; per baseline and metric the mean difference to the
    reference over instances with its sd and SE, the number of instances where the baseline is above, and the
    verdict with the SE over instances (the instance is the unit of replication): 'beats' / 'below' beyond 2 SE."""
    rows = []
    for variant in dict.fromkeys(v["variant"] for v in values):
        x = np.array([[v[m] for m in BASELINE_METRICS] for v in values if v["variant"] == variant])
        rows.append({"variant": variant, "instances": len(x),
                     **{m: float(x[:, i].mean()) for i, m in enumerate(BASELINE_METRICS)},
                     **{f"{m} sd": float(x[:, i].std(ddof=1)) for i, m in enumerate(BASELINE_METRICS)}})
    drows = []
    for baseline in dict.fromkeys(d["baseline"] for d in diffs):
        for m in BASELINE_METRICS:
            x = np.array([d["difference"] for d in diffs if d["baseline"] == baseline and d["metric"] == m])
            mean, sd = float(x.mean()), float(x.std(ddof=1))
            se = sd / np.sqrt(len(x))
            drows.append({"baseline": baseline, "metric": m, "mean difference": mean, "sd": sd, "SE": se,
                          "instances above": int((x > 0).sum()), "instances below": int((x < 0).sum()),
                          "instances": len(x),
                          "vs reference": "beats" if mean > 2 * se else "below" if mean < -2 * se else
                          "matches (within noise)"})
    return rows, drows


def stn_three_panel_figure(data, Z_reference, Z_model, k, title,
                           labels=("classical MDS(64), no learning (L2)", "incidence_mlp s0 embedding (L2)")):
    """STN of the permutation space | of the reference coordinates (plain L2) | of the model embedding for one k:
    the same k centers, node positions (metric MDS of the centers' edge distances), color and size scales; every
    panel title gives n_total, n_best, n_end and best-strength."""
    ref, model = analyse_k(data, Z_reference, k, unit=False), analyse_k(data, Z_model, k)
    fig = plot_stns((ref["G_perm"], ref["G_emb"], model["G_emb"]),
                    (f"permutation space (edge distance), k={k}", f"{labels[0]}, k={k}", f"{labels[1]}, k={k}"),
                    ref["pos"])
    fig.suptitle(title, y=1.03)
    return fig


def generation_r2(data, X):
    """Pooled out-of-fold R^2 of the generation probe (gen_oof) on the features X, not clipped."""
    y = data["generation"].astype(float)
    return float(1 - ((y - gen_oof(X, data)) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def replication(device, instances=REPLICATION_INSTANCES):
    """Per replication instance, on its test runs: criterion and components (gains over the own random init) of the
    replication arch trained on the instance's elites and on its populations, their summary metrics, and the
    difference populations - elites with its bootstrap SE over runs."""
    arch, seed = REPLICATION['arch'], REPLICATION['seed']
    rows = []
    for instance in instances:
        data = drop_repeated_steps(load_or_generate_test_runs(instance))
        models = {f"random init {arch} s{seed}": random_init(arch, device, seed)}
        for objective in REPLICATION['objectives']:
            models[f"{arch} {objective} s{seed}"] = load_or_fit(arch, device, objective=objective, instance=instance,
                                                                seed=seed)
        pieces, summary = evaluate_set(models, data, device)
        W = bootstrap_weights(len(np.unique(data["run_id"])))
        crit = {}
        for objective in REPLICATION['objectives']:
            name = f"{arch} {objective} s{seed}"
            g = gains(pieces, name, W)
            crit[objective] = g.mean(axis=1)
            rows.append({"instance": instance, "variant": objective, "criterion": float(g[0].mean()),
                         "boot SE": float(g[1:].mean(axis=1).std(ddof=1)),
                         **{f"Δ{c}": float(g[0, i]) for i, c in enumerate(COMPONENTS)},
                         **{m: summary[name][m] for m in SUMMARY_METRICS}})
        diff = crit["population_edge"] - crit["edge"]
        rows.append({"instance": instance, "variant": "population_edge - edge", "criterion": float(diff[0]),
                     "boot SE": float(diff[1:].std(ddof=1)), **{f"Δ{c}": float("nan") for c in COMPONENTS},
                     **{m: float("nan") for m in SUMMARY_METRICS}})
    return rows


def across_instances(rows, columns=("criterion",) + tuple(f"Δ{c}" for c in COMPONENTS) + SUMMARY_METRICS):
    """Mean, sd and SE (sd / sqrt(n)) over instances per variant; the instance is the unit of replication."""
    out = []
    for variant in dict.fromkeys(r["variant"] for r in rows):
        rs = [r for r in rows if r["variant"] == variant]
        row = {"variant": variant, "instances": len(rs)}
        for c in columns:
            v = np.array([r[c] for r in rs if c in r], dtype=float)
            if len(v) and not np.isnan(v).all():
                row[c] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "se": float(v.std(ddof=1) / np.sqrt(len(v)))}
        out.append(row)
    return out


def verdict(effect, noise, factor=2.0):
    """'solid' when |effect| exceeds factor times the noise (the largest relevant spread), else 'within noise'."""
    return "solid" if abs(effect) > factor * noise else "within noise"


def v1_history(device):
    """v1 (GA before the fix, best-validation checkpoints): its validation runs and the original trajectory encoder
    (not reoriented) and edge_canonical_100, with the untrained LSTM as their init. History only, not comparable."""
    val_runs, _ = split_runs()
    data = runs_subset(drop_repeated_steps(dict(np.load(os.path.join(V1_DIR, "trajectories.npz")))), val_runs)
    models = {"random init lstm s0": random_init("lstm", device),
              "lstm v1 trajectory s0": load_encoder(os.path.join(V1_DIR, "encoder_variant_trajectory.pt"), device),
              "lstm v1 edge_canonical_100 s0": load_encoder(
                  os.path.join(V1_DIR, "encoder_variant_edge_canonical_100.pt"), device)}
    return data, models


def library_versions():
    names = ["numpy", "scipy", "scikit-learn", "torch", "networkx", "matplotlib", "pyyaml", "nbformat", "nbconvert"]
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    out["cuda (torch build)"] = torch.version.cuda or "none"
    out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return out


def file_hashes(names, directory=OUTPUT_DIR):
    rows = []
    for name in names:
        path = os.path.join(directory, name)
        with open(path, "rb") as f:
            rows.append({"file": name, "bytes": os.path.getsize(path), "sha256": hashlib.sha256(f.read()).hexdigest()})
    return rows
