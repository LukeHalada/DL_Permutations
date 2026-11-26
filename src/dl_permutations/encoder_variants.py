"""Encoder variants that differ only in the training objective, the input orientation and the training length.

Every variant is trained by fit_encoder: same architecture, optimizer, seeds, run split and triplet margin. The
triplets change (TripletDataset: next elite of the run as positive, a tour of another run with a similar length as
negative; EdgeTripletDataset: positive closer to the anchor than the negative in edge distance; LocalEdgeTripletDataset:
the same mixed with triplets around the anchor; SyntheticTripletDataset: perturbed random tours with exact distances),
and so do the input orientation and the number of epochs. Checkpoints go to outputs/encoder_variant_{name}_{hash}.pt;
the original encoder (outputs/contrastive_encoder.pt) is not touched.
"""
import os

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from torch.utils.data import Dataset

from .contrastive_trajectory_learning import (
    DATA_HASH, OUTPUT_DIR, TripletDataset, config, config_hash, drop_repeated_steps, fit_encoder,
    generate_trajectories, make_encoder,
)
from .stn_clusters import STN_N_CENTERS, edge_distance, edge_matrix, embed_batched, full_tours, unit_norm
from .stn_controls import (
    best_vs_size_matched, canonical_orientation, compare_variants, end_equal_chance, probe_scores, tie_ceiling,
)

edge_triplet_config = config['edge_triplet_config']

EDGE_MIN_GAP = edge_triplet_config['min_gap']
PROTOCOL_EPOCHS = config['encoder_protocol_config']['epochs']
NEAR_SHARE = edge_triplet_config['near_share']
NEAR_FRACTION = edge_triplet_config['near_fraction']
SYNTHETIC_SHARE = edge_triplet_config['synthetic_share']
MAX_MOVES = edge_triplet_config['max_moves']
SYNTHETIC_ELITE_SHARE = edge_triplet_config['synthetic_elite_share']
SYNTHETIC_MAX_MOVES = edge_triplet_config['synthetic_max_moves']

# checkpoints depend on the data and on every training setting, trajectories of other instances on the GA settings
MODEL_HASH = config_hash('tsp_config', 'ga_config', 'training_config', 'model_config', 'edge_triplet_config',
                         'encoder_protocol_config')
VARIANT_PATH = os.path.join(OUTPUT_DIR, "encoder_variant_{}_" + MODEL_HASH + ".pt")
INSTANCE_PATH = os.path.join(OUTPUT_DIR, "trajectories_tsp{}_" + DATA_HASH + ".npz")
# results of the GA before the off-by-one fix (tag v1-wyniki), copied there by hand
V1_DIR = os.path.join(OUTPUT_DIR, "v1")


def perturb(tour, rng, max_moves):
    """The tour after 1..max_moves random moves (reverse a segment, i.e. 2-opt, or swap two cities), in canonical
    orientation."""
    tour = tour.copy()
    for _ in range(rng.integers(1, max_moves + 1)):
        i, j = np.sort(rng.choice(len(tour), size=2, replace=False))
        if rng.random() < 0.5:
            tour[i:j + 1] = tour[i:j + 1][::-1]
        else:
            tour[[i, j]] = tour[[j, i]]
    return canonical_orientation(tour)[0]


def pair_edge_distance(a, b):
    """Edge distance between two tours; the same value as edge_distance, faster for a single pair."""
    fa, fb = full_tours(a)[0], full_tours(b)[0]
    adjacent = np.zeros((len(fa), len(fa)), dtype=bool)
    adjacent[fa, np.roll(fa, -1)] = adjacent[np.roll(fa, -1), fa] = True
    return len(fa) - int(adjacent[fb, np.roll(fb, -1)].sum())


def random_position_pairs(rng, m, n):
    """m pairs of different positions lo < hi out of n, every pair equally likely; as columns of shape (m, 1)."""
    i = rng.integers(0, n, size=m)
    j = rng.integers(0, n - 1, size=m)
    j += j >= i
    return np.minimum(i, j)[:, None], np.maximum(i, j)[:, None]


def move_sources(n, lo, hi):
    """Source positions of a 2-opt move (reverse the segment lo..hi) and of a swap (exchange lo and hi), per row."""
    k = np.arange(n)
    return np.where((k >= lo) & (k <= hi), lo + hi - k, k), np.where(k == lo, hi, np.where(k == hi, lo, k))


def perturb_batch(tours, rng, max_moves):
    """perturb for every row of tours (m, n) at once: 1..max_moves random 2-opt moves or swaps per row."""
    m, n = tours.shape
    moves = rng.integers(1, max_moves + 1, size=m)
    for step in range(max_moves):
        reversed_segment, swapped = move_sources(n, *random_position_pairs(rng, m, n))
        source = np.where((rng.random(m) < 0.5)[:, None], reversed_segment, swapped)
        tours = np.take_along_axis(tours, np.where((step < moves)[:, None], source, np.arange(n)), axis=1)
    return canonical_orientation(tours)


def single_move(tours, rng, kind):
    """Every row after exactly one random move of the given kind ("2-opt" or "swap"), not reoriented, and the length
    of the segment between the two positions (hi - lo + 1)."""
    lo, hi = random_position_pairs(rng, *tours.shape)
    reversed_segment, swapped = move_sources(tours.shape[1], lo, hi)
    source = reversed_segment if kind == "2-opt" else swapped
    return np.take_along_axis(tours, source, axis=1), (hi - lo + 1)[:, 0]


def batch_edge_distance(a, b):
    """Edge distance between the tours a[i] and b[i] for every row i."""
    fa, fb = full_tours(a), full_tours(b)
    m, n = fa.shape
    rows = np.arange(m)[:, None]
    next_a, next_b = np.roll(fa, -1, axis=1), np.roll(fb, -1, axis=1)
    adjacent = np.zeros((m, n, n), dtype=bool)
    adjacent[rows, fa, next_a] = adjacent[rows, next_a, fa] = True
    return n - adjacent[rows, fb, next_b].sum(axis=1)


class EdgeTripletDataset(Dataset):
    """Triplets ordered by edge distance: the positive is closer to the anchor than the negative by at least
    min_gap edges.

    Anchor, positive and negative are three different elites drawn uniformly from the given runs, so the triplets
    carry no run or time information, only the order of edge distances. That order is all a nearest-center
    (Voronoi) assignment depends on.
    """

    def __init__(self, data, runs, length, seed, min_gap=None):
        self.perm = data["perm"][np.isin(data["run_id"], runs)]
        self.dist = edge_distance(self.perm, self.perm)
        self.length = length
        self.min_gap = EDGE_MIN_GAP if min_gap is None else min_gap
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.length

    def sample(self):
        while True:
            a, p, n = self.rng.choice(len(self.perm), size=3, replace=False)
            if abs(int(self.dist[a, p]) - int(self.dist[a, n])) >= self.min_gap:
                return (a, p, n) if self.dist[a, p] < self.dist[a, n] else (a, n, p)

    def __getitem__(self, _):
        return tuple(torch.as_tensor(self.perm[i]) for i in self.sample())


class LocalEdgeTripletDataset(EdgeTripletDataset):
    """EdgeTripletDataset mixed with local triplets around the anchor.

    With probability near_share the positive and the negative are both neighbours of the anchor: with probability
    synthetic_share a synthetic one (the anchor after 1..max_moves random 2-opt moves or swaps, in canonical
    orientation, edge distance computed exactly), otherwise one of its nearest elites (near_fraction of the elites
    of the given runs, by edge distance). They are ordered like the global triplets (gap >= min_gap). Tours are
    expected in canonical orientation.
    """

    def __init__(self, data, runs, length, seed, min_gap=None, near_share=None, near_fraction=None,
                 synthetic_share=None, max_moves=None):
        super().__init__(data, runs, length, seed, min_gap)
        self.near_share = NEAR_SHARE if near_share is None else near_share
        self.synthetic_share = SYNTHETIC_SHARE if synthetic_share is None else synthetic_share
        self.max_moves = MAX_MOVES if max_moves is None else max_moves
        n_near = max(2, round((NEAR_FRACTION if near_fraction is None else near_fraction) * len(self.perm)))
        order_dist = self.dist.astype(float)
        np.fill_diagonal(order_dist, -1)
        self.neighbours = np.argsort(order_dist, axis=1, kind="stable")[:, 1:n_near + 1]

    def perturb(self, perm):
        return perturb(perm, self.rng, self.max_moves)

    def neighbour(self, a):
        """A synthetic or real neighbour of elite a and its edge distance to a."""
        if self.rng.random() < self.synthetic_share:
            tour = self.perturb(self.perm[a])
        else:
            tour = self.perm[self.rng.choice(self.neighbours[a])]
        return tour, pair_edge_distance(self.perm[a], tour)

    def sample_local(self):
        a = self.rng.integers(len(self.perm))
        while True:
            (p, dp), (n, dn) = self.neighbour(a), self.neighbour(a)
            if abs(dp - dn) >= self.min_gap:
                return (self.perm[a], p, n) if dp < dn else (self.perm[a], n, p)

    def __getitem__(self, i):
        if self.rng.random() >= self.near_share:
            return super().__getitem__(i)
        return tuple(torch.as_tensor(t) for t in self.sample_local())


class SyntheticTripletDataset(Dataset):
    """Triplets with exactly known edge distances around tours from the whole permutation space.

    The anchor is a uniformly random tour or, with probability elite_share, an elite of the given runs. Positive and
    negative are the anchor after 1..max_moves random 2-opt moves or swaps each, ordered by their exact edge distance
    to the anchor (gap >= min_gap). Every triplet is drawn fresh, so the data are unlimited and only the elite
    admixture comes from the GA. All tours in canonical orientation.

    Triplets are drawn in chunks with numpy. A triplet whose pair fails the gap is dropped together with its anchor;
    the edge distance between a tour and its perturbation depends only on the moved positions, not on the cities, so
    this leaves the distribution of anchors (and the elite share) unchanged.
    """

    CHUNK = 4096

    def __init__(self, data, runs, length, seed, min_gap=None, elite_share=None, max_moves=None):
        self.elites = data["perm"][np.isin(data["run_id"], runs)]
        self.cities = np.sort(data["perm"][0])
        self.length = length
        self.min_gap = EDGE_MIN_GAP if min_gap is None else min_gap
        self.elite_share = SYNTHETIC_ELITE_SHARE if elite_share is None else elite_share
        self.max_moves = SYNTHETIC_MAX_MOVES if max_moves is None else max_moves
        self.rng = np.random.default_rng(seed)
        self.buffer, self.next = np.empty((3, 0, len(self.cities)), dtype=self.elites.dtype), 0

    def __len__(self):
        return self.length

    def anchors(self, m):
        random_tours = canonical_orientation(self.rng.permuted(np.tile(self.cities, (m, 1)), axis=1))
        from_elites = self.rng.random(m) < self.elite_share
        return np.where(from_elites[:, None], self.elites[self.rng.integers(len(self.elites), size=m)], random_tours)

    def sample_chunk(self, m):
        """Accepted triplets (anchor, positive, negative) of m drawn anchors, as an array (3, accepted, n)."""
        anchor = self.anchors(m)
        p, n = perturb_batch(anchor, self.rng, self.max_moves), perturb_batch(anchor, self.rng, self.max_moves)
        dp, dn = batch_edge_distance(anchor, p), batch_edge_distance(anchor, n)
        closer_p = (dp < dn)[:, None]
        keep = np.abs(dp - dn) >= self.min_gap
        return np.stack([anchor, np.where(closer_p, p, n), np.where(closer_p, n, p)])[:, keep]

    def sample(self):
        while self.next >= self.buffer.shape[1]:
            self.buffer, self.next = self.sample_chunk(self.CHUNK), 0
        self.next += 1
        return tuple(self.buffer[:, self.next - 1])

    def __getitem__(self, _):
        return tuple(torch.as_tensor(t) for t in self.sample())


OBJECTIVES = {"trajectory": TripletDataset, "edge": EdgeTripletDataset, "edge_local": LocalEdgeTripletDataset,
              "synthetic": SyntheticTripletDataset}

# fixed number of epochs (encoder_protocol_config), checkpoint of the last epoch, validation only logged
LONG = dict(epochs=PROTOCOL_EPOCHS, fixed_epochs=True)
VARIANTS = {
    "trajectory": dict(objective="trajectory", canonical=False),
    "trajectory_canonical": dict(objective="trajectory", canonical=True),
    "edge_canonical_100": dict(objective="edge", canonical=True, **LONG),
    "edge_local_canonical": dict(objective="edge_local", canonical=True, **LONG),
    "edge_synthetic_canonical": dict(objective="synthetic", canonical=True, **LONG),
}


def variant_inputs(name, perms):
    """Tours as the variant's encoder sees them."""
    return canonical_orientation(perms) if VARIANTS[name]["canonical"] else perms


def load_or_fit_variant(name, data, device):
    """Encoder of a variant; trained with fit_encoder on its triplets and input orientation if no checkpoint exists."""
    spec, path = VARIANTS[name], VARIANT_PATH.format(name)
    if not os.path.exists(path):
        # the checkpoint gets its name only after the last epoch, so an interrupted run leaves no finished-looking file
        partial = path + ".partial"
        inputs = {**data, "perm": variant_inputs(name, data["perm"])}
        model = fit_encoder(inputs, device, dataset=OBJECTIVES[spec["objective"]], path=partial,
                            epochs=spec.get("epochs"), fixed_epochs=spec.get("fixed_epochs", False))
        os.replace(partial, path)
        return model
    return load_encoder(path, device)


def load_encoder(path, device):
    model = make_encoder().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def embed_tours(model, perms, device, canonical=False):
    """Unit-norm embeddings, optionally of the tours in canonical orientation."""
    return unit_norm(embed_batched(model, canonical_orientation(perms) if canonical else perms, device))


def variant_embeddings(name, model, perms, device):
    return embed_tours(model, perms, device, VARIANTS[name]["canonical"])


def runs_subset(data, runs):
    """The rows of the given runs; run grouping and generation order are kept."""
    mask = np.isin(data["run_id"], runs)
    return {key: value[mask] for key, value in data.items()}


def distance_correlation(Z, perms):
    """Spearman correlation between embedding (L2) and edge distances over all pairs of elites."""
    i, j = np.triu_indices(len(perms), 1)
    return float(spearmanr(cdist(Z, Z)[i, j], edge_distance(perms, perms)[i, j])[0])


def distinct_cycles(perms):
    """Indices of the first occurrence of every distinct cycle, in data order."""
    return np.sort(np.unique(edge_matrix(perms), axis=0, return_index=True)[1])


def local_metrics(Z, perms, ks=(5, 10)):
    """recall@k on distinct cycles: share of the embedding's k nearest tours that lie within the k-th smallest edge
    distance of the tour (edge distances are integers with many ties, so every tour at that distance counts)."""
    idx = distinct_cycles(perms)
    D = edge_distance(perms[idx], perms[idx]).astype(float)
    E = cdist(Z[idx], Z[idx])
    np.fill_diagonal(D, np.inf)
    np.fill_diagonal(E, np.inf)
    nearest = np.argsort(E, axis=1)
    return {f"recall@{k}": float((np.take_along_axis(D, nearest[:, :k], axis=1) <= np.sort(D, axis=1)[:, k - 1:k])
                                 .mean()) for k in ks}


def local_chance(perms, ks=(5, 10)):
    """recall@k of a random ranking: mean share of tours within the k-th smallest edge distance (distinct cycles)."""
    idx = distinct_cycles(perms)
    D = edge_distance(perms[idx], perms[idx]).astype(float)
    np.fill_diagonal(D, np.inf)
    out = {f"recall@{k}": float(((D <= np.sort(D, axis=1)[:, k - 1:k]).sum(axis=1) / (len(idx) - 1)).mean())
           for k in ks}
    out["distinct_cycles"] = len(idx)
    return out


MOVE_KINDS = ("swap", "2-opt")
SEGMENT_BINS = ((2, 2), (3, 6), (7, 12), (13, 19))


def move_neighbour_ranks(embed, perms, seed=0, per_anchor=3):
    """Single-move neighbours of every distinct cycle (anchor) by move kind, and their rank among the other cycles.

    For each anchor and kind, per_anchor neighbours after exactly one random move (the same neighbours for every
    encoder with the same seed); neighbours that are the anchor's own cycle (2-opt of the whole tour) are dropped.
    emb_rank / edge_rank: number of other distinct cycles strictly closer to the anchor than the neighbour in the
    embedding / in edge distance, so the neighbour is among the k nearest when the rank is < k. embed maps tours to
    unit embeddings with the encoder's input orientation.
    """
    rng = np.random.default_rng(seed)
    E = perms[distinct_cycles(perms)]
    Z = embed(E)
    D_emb, D_edge = cdist(Z, Z), edge_distance(E, E).astype(float)
    i, j = np.triu_indices(len(E), 1)
    median_pair = np.median(D_emb[i, j])
    np.fill_diagonal(D_emb, np.inf)
    np.fill_diagonal(D_edge, np.inf)
    anchors = np.repeat(np.arange(len(E)), per_anchor)
    out = {}
    for kind in MOVE_KINDS:
        T, segment = single_move(E[anchors], rng, kind)
        d_edge = batch_edge_distance(E[anchors], T)
        keep = d_edge > 0
        a, T, segment, d_edge = anchors[keep], T[keep], segment[keep], d_edge[keep]
        d_emb = np.linalg.norm(Z[a] - embed(T), axis=1)
        out[kind] = {"segment": segment, "d_edge": d_edge, "d_emb_ratio": d_emb / median_pair,
                     "emb_rank": (D_emb[a] < d_emb[:, None]).sum(axis=1),
                     "edge_rank": (D_edge[a] < d_edge[:, None]).sum(axis=1)}
    return out


def move_recall_tables(ranks, ks=(1, 5, 10)):
    """Tables of move_neighbour_ranks results keyed by encoder: recall@k of the neighbour per move kind (with its
    edge distance, the same recall in edge distance and the median embedding distance relative to the median pair
    distance), and recall@5 by the length of the moved segment."""
    overall, by_segment = [], []
    for name, by_kind in ranks.items():
        for kind, r in by_kind.items():
            overall.append({"encoder": name, "move": kind, "n": len(r["d_edge"]), "d_edge": float(r["d_edge"].mean()),
                            **{f"recall@{k}": float((r["emb_rank"] < k).mean()) for k in ks},
                            "edge recall@1": float((r["edge_rank"] < 1).mean()),
                            "d_emb / median": float(np.median(r["d_emb_ratio"]))})
            row = {"encoder": name, "move": kind}
            for lo, hi in SEGMENT_BINS:
                m = (r["segment"] >= lo) & (r["segment"] <= hi)
                row[f"seg {lo}-{hi}" if lo < hi else f"seg {lo}"] = float((r["emb_rank"][m] < 5).mean()) if m.any() \
                    else float("nan")
            by_segment.append(row)
    return {"single-move neighbours: recall@k": overall, "recall@5 by segment length (positions lo..hi)": by_segment}


def load_or_generate_instance(tsp_seed):
    """Elite trajectories of the same GA (same run seeds) on another TSP instance, cached per instance."""
    path = INSTANCE_PATH.format(tsp_seed)
    if os.path.exists(path):
        return dict(np.load(path))
    data = generate_trajectories(tsp_seed)
    np.savez_compressed(path, **data)
    return data


def load_v1(names, device):
    """v1 results (GA before the fix): elite trajectories without repeated steps and the variant encoders."""
    data = drop_repeated_steps(dict(np.load(os.path.join(V1_DIR, "trajectories.npz"))))
    return data, {name: load_encoder(os.path.join(V1_DIR, f"encoder_variant_{name}.pt"), device) for name in names}


def pivot(rows, metric, key="variant", name=None):
    """One row per encoder (key) with the metric per k as columns k=5, k=10, ...; name renames every row."""
    table = {}
    for r in rows:
        table.setdefault(r[key], {"encoder": name or r[key]})[f"k={r['k']}"] = r[metric]
    return list(table.values())


GEOMETRY_COLUMNS = ["encoder", "recall@5", "recall@10", "spearman", "rev_ratio", "run_acc", "gen_r2"]


def geometry_rows(Z, data, Z_rev=None):
    """Per encoder: recall@k, Spearman of embedding and edge distances over all pairs, the distance to the own
    reversal relative to the median pair distance (when Z_rev is given) and the linear probes; then the permutation
    space itself (edge incidence) and chance. Returns the rows and local_chance."""
    perms, nan = data["perm"], float("nan")
    chance = local_chance(perms)
    rows = []
    for name, Zn in Z.items():
        rev = nan
        if Z_rev is not None:
            i, j = np.triu_indices(len(Zn), 1)
            rev = float(np.median(np.linalg.norm(Zn - Z_rev[name], axis=1)) / np.median(cdist(Zn, Zn)[i, j]))
        rows.append({"encoder": name, **local_metrics(Zn, perms), "spearman": distance_correlation(Zn, perms),
                     "rev_ratio": rev, **probe_scores(Zn, data)})
    rows.append({"encoder": "edge incidence (reference)", "recall@5": 1.0, "recall@10": 1.0, "spearman": 1.0,
                 "rev_ratio": 0.0, **probe_scores(edge_matrix(perms), data)})
    rows.append({"encoder": "chance", "recall@5": chance["recall@5"], "recall@10": chance["recall@10"],
                 "spearman": nan, "rev_ratio": nan, **probe_scores(np.zeros((len(perms), 1)), data)})
    return rows, chance


def stn_pivots(data, Z, val_runs, ks=STN_N_CENTERS):
    """STN comparison of every embedding with the permutation space as tables with one row per encoder and one
    column per k: ARI and AMI (with the tie-breaking ceiling), end_equal, n_end and best_strength (with the
    permutation-space row) and the share of tied elites. Returns the tables and the analyse_k results keyed by
    (encoder, k)."""
    rows, results = compare_variants(data, Z, val_runs, ks)
    ceiling = [tie_ceiling(data, k) for k in ks]
    perm = [r for r in rows if r["variant"] == rows[0]["variant"]]
    tables = {m: pivot(rows, m) + [{"encoder": "ceiling (perm, random ties)", **{f"k={c['k']}": c[m] for c in ceiling}}]
              for m in ("ari", "ami")}
    tables["end_equal"] = pivot(rows, "end_equal")
    for m in ("n_end", "best_strength"):
        tables[m] = pivot(perm, f"{m}_perm", name="permutation space") + pivot(rows, f"{m}_emb")
    tables["ties"] = pivot(perm, "ties", name="permutation space")
    return tables, results


def funnel_pivots(results, data):
    """Size-matched null for best_strength and end_equal against its chance level, one column per k.

    best_strength: observed vs. the mean entry share of locations of similar size [share of them the best location
    beats], for the permutation space and every embedding; size: share of elites in the best location and the range
    of the reference locations; end_equal: observed / chance (sum over locations of p_perm * p_emb of final
    locations).
    """
    run_id, fitness = data["run_id"], data["fitness"]
    funnel, size, ends = [], [], []
    first = next(iter(results))[0]
    for (name, k), res in results.items():
        spaces = ((("permutation space", res["perm_labels"]),) if name == first else ()) + ((name, res["emb_labels"]),)
        for space, labels in spaces:
            m = best_vs_size_matched(labels, run_id, fitness)
            funnel.append({"space": space, "k": k,
                           "cell": f"{m['best_strength']:.2f} vs {m['ref_mean']:.2f} [{m['beats']:.2f}]"})
            size.append({"space": space, "k": k,
                         "cell": f"{m['best_share']:.2f} vs {m['ref_share_min']:.2f}-{m['ref_share_max']:.2f}"})
        last = np.r_[run_id[1:] != run_id[:-1], True]
        ends.append({"space": name, "k": k,
                     "cell": f"{np.mean(res['perm_labels'][last] == res['emb_labels'][last]):.2f} / "
                             f"{end_equal_chance(res['perm_labels'], res['emb_labels'], run_id):.2f}"})
    return {"best_strength vs size-matched locations [share beaten]": pivot(funnel, "cell", key="space"),
            "share of elites: best location vs references": pivot(size, "cell", key="space"),
            "end_equal / chance": pivot(ends, "cell", key="space")}


def robustness_tables(tables_v1, tables_v2, metrics=("ties", "n_end", "best_strength", "ari", "end_equal")):
    """v1 and v2 rows of the same stn_pivots tables next to each other, per metric."""
    out = {}
    for m in metrics:
        rows = []
        for row_v1, row_v2 in zip(tables_v1[m], tables_v2[m]):
            rows += [{**row_v1, "encoder": f"{row_v1['encoder']} (v1)"}, {**row_v2, "encoder": f"{row_v2['encoder']} (v2)"}]
        out[m] = rows
    return out
