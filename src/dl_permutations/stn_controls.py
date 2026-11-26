"""Controls for the STN comparison of permutation and embedding space.

Orientation (the encoder reads a sequence, the edge distance does not), an untrained encoder as a baseline,
linear probes for run identity and generation, a null model for best-strength and the sensitivity of the
permutation-space partition to how ties between centers are broken.
"""
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .contrastive_trajectory_learning import START_CITY, TORCH_SEED, make_encoder
from .stn_clusters import (
    analyse_k, build_stn, edge_distance, edge_matrix, embed_batched, embedding_distance, nearest_center,
    sample_centers, stn_metrics, summary_row, unit_norm,
)

CLOSE_EDGE_DISTANCE = 4
NULL_PERMUTATIONS = 1000
TIE_SEEDS = range(10)
PROBE_FOLDS = 5


def reverse_tours(perms):
    """The same cycles traversed in the opposite direction; city 0 stays fixed at the start."""
    assert START_CITY == 1, "perm[::-1] is the reversed cycle only with exactly one fixed start city"
    return np.ascontiguousarray(np.atleast_2d(perms)[:, ::-1])


def canonical_orientation(perms):
    """Each tour in the direction that leaves city 0 towards its smaller neighbour, i.e. perm[0] < perm[-1].

    Both orientations of a cycle map to the same sequence, so the encoder gets one input per cycle.
    """
    perms = np.atleast_2d(perms)
    return np.where((perms[:, 0] > perms[:, -1])[:, None], reverse_tours(perms), perms)


def random_init_encoder(device):
    """Untrained baseline: the encoder at the initialization fit_encoder starts from (same torch seed)."""
    torch.manual_seed(TORCH_SEED)
    return make_encoder().to(device)


def unit_embeddings(model, perms, device):
    return unit_norm(embed_batched(model, perms, device))


def reversal_report(Z, Z_rev, data, close=CLOSE_EDGE_DISTANCE):
    """Embedding distance of each elite to its own reversal (same cycle, edge distance 0) against pairs of
    distinct tours: consecutive elites of a run, tours at edge distance <= close, and all pairs.

    Returns one row per group (pair count, median edge distance, quartiles of the embedding distance) and a
    summary: the best tour's distance to its reversal, the median embedding distance from the best tour to its
    close tours, and the share of distinct tours lying closer to a tour than its own reversal (best tour and
    median over elites).
    """
    perms, run_id = data["perm"], data["run_id"]
    D_edge = edge_distance(perms, perms)
    D_emb = cdist(Z, Z)
    d_rev = np.linalg.norm(Z - Z_rev, axis=1)
    i, j = np.triu_indices(len(perms), 1)
    flipped = (D_edge[i, j] == 0) & (perms[i] != perms[j]).any(axis=1)
    distinct = D_edge[i, j] > 0
    near = distinct & (D_edge[i, j] <= close)
    nxt = np.flatnonzero(run_id[1:] == run_id[:-1])
    groups = {
        "own reversal": (np.zeros(len(perms)), d_rev),
        "same cycle stored reversed": (D_edge[i, j][flipped], D_emb[i, j][flipped]),
        "consecutive elites of a run": (D_edge[nxt, nxt + 1], D_emb[nxt, nxt + 1]),
        f"edge distance 1-{close}": (D_edge[i, j][near], D_emb[i, j][near]),
        "all distinct pairs": (D_edge[i, j][distinct], D_emb[i, j][distinct]),
    }
    rows = [{"pairs": name, "n": len(e), "edge_median": float(np.median(e)) if len(e) else np.nan,
             **dict(zip(("emb_q25", "emb_median", "emb_q75"),
                        map(float, np.percentile(d, [25, 50, 75]) if len(d) else [np.nan] * 3)))}
            for name, (e, d) in groups.items()]

    other = D_edge > 0
    closer = ((D_emb < d_rev[:, None]) & other).sum(axis=1) / other.sum(axis=1)
    best = int(np.argmin(data["fitness"]))
    best_close = other[best] & (D_edge[best] <= close)
    summary = {
        "best_to_reversal": float(d_rev[best]),
        "best_close_tours": int(best_close.sum()),
        "best_to_close_median": float(np.median(D_emb[best, best_close])) if best_close.any() else np.nan,
        "best_closer_than_reversal": float(closer[best]),
        "median_closer_than_reversal": float(np.median(closer)),
    }
    return rows, summary


def compare_variants(data, variants, val_runs, ks):
    """Agreement with the permutation-space partition and the embedding-space STN for several embeddings of
    the same elites (same centers, same permutation-space labels). Returns summary rows tagged with the
    variant name and the analyse_k results keyed by (variant, k)."""
    rows, results = [], {}
    for name, Z in variants.items():
        for k in ks:
            res = results[name, k] = analyse_k(data, Z, k)
            rows.append({"variant": name, **summary_row(res, data, Z, val_runs)})
    return rows, results


def location_entries(labels, run_id):
    """Which runs enter which location (at least one transition into it, as in build_stn) and the share of elites
    per location: a (runs, locations) bool matrix and a vector. Rows must be grouped by run and ordered by
    generation."""
    runs, run_idx = np.unique(run_id, return_inverse=True)
    moved = np.r_[False, (labels[1:] != labels[:-1]) & (run_idx[1:] == run_idx[:-1])]
    entered = np.zeros((len(runs), labels.max() + 1), dtype=bool)
    entered[run_idx[moved], labels[moved]] = True
    return entered, np.bincount(labels, minlength=entered.shape[1]) / len(labels)


def best_vs_size_matched(labels, run_id, fitness, factor=2.0, min_ref=3):
    """best_strength against locations of similar size: a null model for the funnel that keeps the run dynamics.

    The best location(s) (cells holding the best tour) are compared with the other occupied locations whose share
    of elites lies within a factor of the best location's share; if fewer than min_ref qualify, the min_ref closest
    in log share are used. Returns best_strength (share of runs entering a best location) and the best share of
    elites, the mean and maximum entry share of the reference locations, their number and share range, and the
    fraction of references the best location beats (ties count half).
    """
    entered, share = location_entries(labels, run_id)
    min_fitness = np.full(len(share), np.inf)
    np.minimum.at(min_fitness, labels, fitness)
    best = np.isclose(min_fitness, fitness.min())
    best_strength, best_share = entered[:, best].any(axis=1).mean(), share[best].sum()
    others = np.flatnonzero((share > 0) & ~best)
    gap = np.abs(np.log(share[others] / best_share))
    refs = others[gap <= np.log(factor)]
    if len(refs) < min_ref:
        refs = others[np.argsort(gap, kind="stable")[:min_ref]]
    ref_entry = entered[:, refs].mean(axis=0)
    return {
        "best_strength": float(best_strength),
        "best_share": float(best_share),
        "ref_mean": float(ref_entry.mean()),
        "ref_max": float(ref_entry.max()),
        "n_ref": len(refs),
        "ref_share_min": float(share[refs].min()),
        "ref_share_max": float(share[refs].max()),
        "beats": float(np.mean(ref_entry < best_strength) + 0.5 * np.mean(ref_entry == best_strength)),
    }


def end_equal_chance(labels_a, labels_b, run_id):
    """Chance level of end_equal: probability that independent final locations coincide given the distributions of
    final locations in both spaces, sum_v p_a(v) p_b(v) (sum p^2 when the two distributions are equal)."""
    last = np.r_[run_id[1:] != run_id[:-1], True]
    size = max(labels_a.max(), labels_b.max()) + 1
    p_a = np.bincount(labels_a[last], minlength=size) / last.sum()
    p_b = np.bincount(labels_b[last], minlength=size) / last.sum()
    return float(p_a @ p_b)


def best_strength_null(labels, run_id, fitness, n_perm=NULL_PERMUTATIONS, seed=0):
    """Best-strength under label shuffling that keeps the size of every cell (superseded by best_vs_size_matched:
    the shuffle destroys the runs' autocorrelation, so every space falls below it; kept for the v1 notebooks).

    Labels are permuted across all elites; the best node(s) stay those of the observed STN (cells holding the
    global best tour), so their size is preserved as well. A run counts if it has at least one transition into
    a best node, as in build_stn. Shuffling also destroys the temporal autocorrelation of the runs, so shuffled
    runs change location far more often; the mean number of transitions per run is returned for both.
    Rows must be grouped by run and ordered by generation, as in the trajectories.
    """
    min_fitness = np.full(labels.max() + 1, np.inf)
    np.minimum.at(min_fitness, labels, fitness)
    best = np.isclose(min_fitness, fitness.min())
    runs, run_idx = np.unique(run_id, return_inverse=True)
    same_run = run_idx[1:] == run_idx[:-1]
    run_of_step = np.eye(len(runs), dtype=np.float32)[run_idx[1:]]

    def stats(L):
        moves = (L[:, 1:] != L[:, :-1]) & same_run
        entered = ((moves & best[L[:, 1:]]).astype(np.float32) @ run_of_step) > 0
        return entered.mean(axis=1), moves.sum(axis=1) / len(runs)

    rng = np.random.default_rng(seed)
    (observed,), (moves_observed,) = stats(labels[None])
    null, moves_null = stats(np.stack([rng.permutation(labels) for _ in range(n_perm)]))
    return {
        "best_share": float(best[labels].mean()),
        "observed": float(observed),
        "null_mean": float(null.mean()),
        "null_q025": float(np.quantile(null, 0.025)),
        "null_q975": float(np.quantile(null, 0.975)),
        "percentile": float(np.mean(null < observed) + 0.5 * np.mean(null == observed)),
        "moves_per_run": float(moves_observed),
        "moves_per_run_null": float(moves_null.mean()),
    }


def null_rows(results, spaces, data, n_perm=NULL_PERMUTATIONS):
    """best_strength_null for the permutation-space labels and each embedding variant, per k.

    spaces maps a row name to (variant, label key), e.g. {"perm": ("trained", "perm_labels")}."""
    rows = [{"space": name, "k": k, **best_strength_null(res[key], data["run_id"], data["fitness"], n_perm)}
            for (variant, k), res in results.items()
            for name, (v, key) in spaces.items() if v == variant]
    return sorted(rows, key=lambda r: r["k"])


def nearest_center_random_ties(dist, cycle_id, seed):
    """Nearest center with ties broken uniformly at random, the same way for all copies of one cycle.

    dist holds integer edge distances, so noise below 1 only reorders tied centers.
    """
    noise = np.random.default_rng(seed).random((cycle_id.max() + 1, dist.shape[1]))
    return (dist + 0.5 * noise[cycle_id]).argmin(axis=1)


def tie_spread(data, Z, k, seeds=TIE_SEEDS):
    """Permutation-space STN metrics and agreement with the embedding partition when ties are broken at
    random (one labeling per seed) instead of by the lowest center index.

    Returns the values under the default rule and arrays over the seeds, plus the share of tied elites and the
    mean share of elites whose label differs from the default.
    """
    perms, run_id, fitness = data["perm"], data["run_id"], data["fitness"]
    centers = sample_centers(perms, k)
    D = edge_distance(perms, perms[centers])
    cycle_id = np.unique(edge_matrix(perms), axis=0, return_inverse=True)[1].ravel()
    emb_labels, _ = nearest_center(embedding_distance(Z, Z[centers]))
    default, ties = nearest_center(D)

    def metrics(labels):
        m = stn_metrics(build_stn(labels, run_id, fitness))
        return {"ari": adjusted_rand_score(labels, emb_labels), "ami": adjusted_mutual_info_score(labels, emb_labels),
                "n_end": m["n_end"], "best_strength": m["best_strength"]}

    randomized = [nearest_center_random_ties(D, cycle_id, s) for s in seeds]
    per_seed = [metrics(labels) for labels in randomized]
    return {
        "k": k,
        "ties": float(ties.mean()),
        "relabeled": float(np.mean([np.mean(labels != default) for labels in randomized])),
        "default": metrics(default),
        "seeds": {m: np.array([r[m] for r in per_seed]) for m in per_seed[0]},
    }


def tie_rows(spreads):
    """One row per k: share of tied elites, mean share relabeled, and each metric as default [min, max]."""
    def cell(default, values):
        fmt = "{:.0f}" if isinstance(default, (int, np.integer)) else "{:.3f}"
        return f"{fmt.format(default)} [{fmt.format(values.min())}, {fmt.format(values.max())}]"
    return [{"k": s["k"], "ties": s["ties"], "relabeled": s["relabeled"],
             **{m: cell(s["default"][m], s["seeds"][m]) for m in s["default"]}} for s in spreads]


def tie_ceiling(data, k, seeds=TIE_SEEDS):
    """Agreement of the permutation-space partition with itself when only the tie-breaking changes: mean ARI and
    AMI between the lowest-index labels and random tie-breaking. A practical upper reference for the agreement of
    an embedding partition with the permutation space."""
    perms = data["perm"]
    D = edge_distance(perms, perms[sample_centers(perms, k)])
    cycle_id = np.unique(edge_matrix(perms), axis=0, return_inverse=True)[1].ravel()
    default, _ = nearest_center(D)
    randomized = [nearest_center_random_ties(D, cycle_id, s) for s in seeds]
    return {"k": k, "ari": float(np.mean([adjusted_rand_score(default, r) for r in randomized])),
            "ami": float(np.mean([adjusted_mutual_info_score(default, r) for r in randomized]))}


def probe_scores(X, data, folds=PROBE_FOLDS, seed=0):
    """Linear probes on standardized features: accuracy for run_id (stratified folds, every run is seen in
    training) and R^2 for the generation (folds grouped by run, so it is tested on unseen runs)."""
    runs, counts = np.unique(data["run_id"], return_counts=True)
    run_folds, group_folds = min(folds, counts.min()), min(folds, len(runs))
    run_acc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                              X, data["run_id"], cv=StratifiedKFold(run_folds, shuffle=True, random_state=seed))
    gen_r2 = cross_val_score(make_pipeline(StandardScaler(), Ridge()), X, data["generation"],
                             cv=GroupKFold(group_folds), groups=data["run_id"], scoring="r2")
    return {"run_acc": float(run_acc.mean()), "gen_r2": float(gen_r2.mean())}


def probe_rows(feature_sets, data):
    return [{"features": name, "dims": X.shape[1], **probe_scores(X, data)} for name, X in feature_sets.items()]
