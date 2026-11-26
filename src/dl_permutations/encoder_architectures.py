"""Encoder architectures under one training protocol, data variants on top of them, and the evaluation sets.

Protocol (encoder_protocol_config): fit_encoder with fixed_epochs=True, i.e. exactly `epochs` epochs, checkpoint of the
last epoch, validation triplets only logged. A seed shifts the initialization and the training triplets; for a given
seed and objective every architecture sees the same triplets, the same run split and the same margin. Inputs are tours
in canonical orientation (only the LSTM depends on it). Checkpoints: outputs/encoder_{arch}_{objective}_{instance}_
s{seed}_{hash}.pt, the per-epoch log next to them (.json); a checkpoint gets its name only after the last epoch.

Architectures: the LSTM of the original pipeline (RNNEncoder), an MLP on the edge incidence vector of the full cycle
(IncidenceMLPEncoder) and DeepSets over its undirected edges (EdgeDeepSetsEncoder), with the LSTM's sizes; the last two
are invariant to the direction and rotation of the cycle by construction.

Objectives: "edge" (EdgeTripletDataset: three elites ordered by edge distance), "elite_synthetic" (SyntheticTriplet-
Dataset with anchors only from elites, positive and negative its synthetic neighbours), "population_edge" (as "edge"
on every distinct tour of the GA populations of a run). Instances: "tsp21" (instance 1) and "tsp99" (instance 2).

Local term (local_term_config): optionally the triplet loss of (elite, the elite after one swap or 2-opt move, one
of its nearest elites) is added with a weight; checkpoints carry the weight and a hash of the section in their names.

Evaluation sets: validation runs of instance 1, a test set of new GA runs on instance 1 (test_set_config, never used
for training or selection), instance 2 (all runs, or its validation runs for encoders trained on instance 2). Training
is per instance: no encoder trained on one instance transfers to another (the city and edge identities it learns are
specific to the instance). Tables report every trained encoder (mean±sd over seeds) with its gain over its own random
init (same architecture and seed) and, as a second number, the difference to the untrained LSTM.
"""
import json
import os
import re
from functools import lru_cache, partial

import numpy as np
import torch
from scipy.stats import rankdata
from torch.utils.data import Dataset

from .contrastive_trajectory_learning import (
    CITIES_COUNT, DATA_HASH, DROPOUT, EMB_DIM, HIDDEN_SIZE, N_RUNS, OUT_DIM, OUTPUT_DIR, START_CITY, TORCH_SEED, config,
    config_hash, drop_repeated_steps, fit_encoder, generate_trajectories, load_or_generate_trajectories, make_encoder, split_runs,
)
from .encoder_variants import (
    EDGE_MIN_GAP, MODEL_HASH, PROTOCOL_EPOCHS, EdgeTripletDataset, SyntheticTripletDataset, batch_edge_distance,
    embed_tours, geometry_rows, load_or_generate_instance, runs_subset, single_move, stn_pivots,
)
from .models.edge_deepsets_encoder import EdgeDeepSetsEncoder
from .models.incidence_mlp_encoder import IncidenceMLPEncoder
from .stn_clusters import (
    STN_CENTERS_SEED, STN_N_CENTERS, edge_distance, embedding_distance, generate_populations,
    load_or_generate_populations, sample_centers,
)
from .stn_controls import canonical_orientation, end_equal_chance

PROTOCOL_SEEDS = config['encoder_protocol_config']['seeds']
TEST_RUNS = config['test_set_config']['n_runs']
TEST_PATH = os.path.join(OUTPUT_DIR, f"trajectories_test{TEST_RUNS}_{DATA_HASH}.npz")
INSTANCE_TEST_PATH = os.path.join(OUTPUT_DIR, "trajectories_test" + str(TEST_RUNS) + "_tsp{}_" + DATA_HASH + ".npz")
DEV_FIRST_RUN = config['dev_set_config']['first_run']
DEV_RUNS = config['dev_set_config']['n_runs']
DEV_PATH = os.path.join(OUTPUT_DIR, f"trajectories_dev{DEV_RUNS}_r{DEV_FIRST_RUN}_{DATA_HASH}.npz")
INSTANCE_POPULATIONS_PATH = os.path.join(OUTPUT_DIR, "populations_tsp{}_" + DATA_HASH + ".npz")
REPLICATION = config['replication_config']
ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder_{arch}_{objective}_{instance}_s{seed}_" + MODEL_HASH + ".pt")
local_term_config = config['local_term_config']
LOCAL_NEAR_FRACTION = local_term_config['near_fraction']
LOCAL_WEIGHTS = local_term_config['weights']
LOCAL_HASH = config_hash('local_term_config')
INSTANCE_SEEDS = {"tsp21": None, "tsp99": 99, **{f"tsp{seed}": seed for seed in REPLICATION['tsp_seeds']}}
REFERENCE = "random init lstm s0"


def make_incidence_mlp():
    return IncidenceMLPEncoder(CITIES_COUNT, START_CITY, HIDDEN_SIZE, OUT_DIM, DROPOUT, normalize=True)


def make_edge_deepsets():
    return EdgeDeepSetsEncoder(CITIES_COUNT, START_CITY, EMB_DIM, HIDDEN_SIZE, OUT_DIM, DROPOUT, normalize=True)


ARCHITECTURES = {"lstm": make_encoder, "incidence_mlp": make_incidence_mlp, "edge_deepsets": make_edge_deepsets}


class PoolEdgeTripletDataset(Dataset):
    """EdgeTripletDataset for pools too large for a distance matrix (whole populations).

    Three different tours of the given runs, accepted when their edge distances to the anchor differ by >= min_gap
    and ordered by them; drawn in numpy chunks with exact distances. Both datasets redraw the whole triplet on
    rejection, so the distribution is the same.
    """

    CHUNK = 4096

    def __init__(self, data, runs, length, seed, min_gap=None):
        self.perm = data["perm"][np.isin(data["run_id"], runs)]
        self.length = length
        self.min_gap = EDGE_MIN_GAP if min_gap is None else min_gap
        self.rng = np.random.default_rng(seed)
        self.buffer, self.next = np.empty((3, 0, self.perm.shape[1]), dtype=self.perm.dtype), 0

    def __len__(self):
        return self.length

    def sample_chunk(self, m):
        idx = self.rng.integers(0, len(self.perm), size=(m, 3))
        a, p, n = (self.perm[idx[:, c]] for c in range(3))
        dp, dn = batch_edge_distance(a, p), batch_edge_distance(a, n)
        distinct = (idx[:, 0] != idx[:, 1]) & (idx[:, 0] != idx[:, 2]) & (idx[:, 1] != idx[:, 2])
        keep = distinct & (np.abs(dp - dn) >= self.min_gap)
        closer_p = (dp < dn)[:, None]
        return np.stack([a, np.where(closer_p, p, n), np.where(closer_p, n, p)])[:, keep]

    def sample(self):
        while self.next >= self.buffer.shape[1]:
            self.buffer, self.next = self.sample_chunk(self.CHUNK), 0
        self.next += 1
        return tuple(self.buffer[:, self.next - 1])

    def __getitem__(self, _):
        return tuple(torch.as_tensor(t) for t in self.sample())


class LocalMoveTripletDataset(Dataset):
    """Triplets of the local term: the anchor is an elite of the given runs, the positive the anchor after exactly one
    random move (swap or 2-opt, equally likely), the negative one of the anchor's nearest elites (near_fraction of the
    elites by edge distance); accepted when the negative is farther than the positive by at least min_gap edges
    (exact distances). Drawn in numpy chunks; tours in canonical orientation."""

    CHUNK = 4096

    def __init__(self, data, runs, length, seed, min_gap=None, near_fraction=None):
        self.perm = data["perm"][np.isin(data["run_id"], runs)]
        self.length = length
        self.min_gap = EDGE_MIN_GAP if min_gap is None else min_gap
        n_near = max(2, round((LOCAL_NEAR_FRACTION if near_fraction is None else near_fraction) * len(self.perm)))
        dist = edge_distance(self.perm, self.perm).astype(float)
        np.fill_diagonal(dist, np.inf)
        self.near = np.argsort(dist, axis=1, kind="stable")[:, :n_near]
        self.rng = np.random.default_rng(seed)
        self.buffer, self.next = np.empty((3, 0, self.perm.shape[1]), dtype=self.perm.dtype), 0

    def __len__(self):
        return self.length

    def sample_chunk(self, m):
        idx = self.rng.integers(len(self.perm), size=m)
        anchor = self.perm[idx]
        swapped, _ = single_move(anchor, self.rng, "swap")
        reversed_segment, _ = single_move(anchor, self.rng, "2-opt")
        positive = canonical_orientation(np.where((self.rng.random(m) < 0.5)[:, None], reversed_segment, swapped))
        negative = self.perm[self.near[idx, self.rng.integers(self.near.shape[1], size=m)]]
        dp, dn = batch_edge_distance(anchor, positive), batch_edge_distance(anchor, negative)
        keep = (dp > 0) & (dn - dp >= self.min_gap)
        return np.stack([anchor, positive, negative])[:, keep]

    def sample(self):
        while self.next >= self.buffer.shape[1]:
            self.buffer, self.next = self.sample_chunk(self.CHUNK), 0
        self.next += 1
        return tuple(self.buffer[:, self.next - 1])

    def __getitem__(self, _):
        return tuple(torch.as_tensor(t) for t in self.sample())


OBJECTIVES = {"edge": EdgeTripletDataset, "elite_synthetic": partial(SyntheticTripletDataset, elite_share=1.0),
              "population_edge": PoolEdgeTripletDataset}


def instance_elites(instance):
    """Elite trajectories without repeated steps of an instance."""
    raw = load_or_generate_trajectories() if instance == "tsp21" else load_or_generate_instance(INSTANCE_SEEDS[instance])
    return drop_repeated_steps(raw)


def instance_populations(instance):
    """GA populations of every training run of an instance, shape (runs, generations, pop, len); cached per instance."""
    if instance == "tsp21":
        return load_or_generate_populations()
    path = INSTANCE_POPULATIONS_PATH.format(INSTANCE_SEEDS[instance])
    if os.path.exists(path):
        return np.load(path)["populations"]
    populations = generate_populations(INSTANCE_SEEDS[instance])
    np.savez_compressed(path, populations=populations)
    return populations


@lru_cache(maxsize=None)
def population_tours(instance="tsp21"):
    """Every distinct cycle in the GA populations of each run of an instance (canonical orientation), with its run."""
    populations = instance_populations(instance)
    perms, run_ids = [], []
    for run, run_populations in enumerate(populations):
        tours = np.unique(canonical_orientation(run_populations.reshape(-1, run_populations.shape[-1])), axis=0)
        perms.append(tours)
        run_ids.append(np.full(len(tours), run))
    return {"perm": np.concatenate(perms), "run_id": np.concatenate(run_ids)}


def training_data(objective, instance):
    if objective == "population_edge":
        return population_tours(instance)
    data = instance_elites(instance)
    return {**data, "perm": canonical_orientation(data["perm"])}


def checkpoint_path(arch, objective="edge", instance="tsp21", seed=0, local_weight=0.0):
    """Checkpoint of a protocol run; with a local term its weight and a hash of local_term_config join the name."""
    if local_weight:
        objective = f"{objective}+local{local_weight:g}-{LOCAL_HASH}"
    return ENCODER_PATH.format(arch=arch, objective=objective, instance=instance, seed=seed)


def history_path(path):
    return os.path.splitext(path)[0] + ".json"


def load_model(arch, path, device):
    model = ARCHITECTURES[arch]().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def load_or_fit(arch, device, objective="edge", instance="tsp21", seed=0, local_weight=0.0):
    """Encoder trained under the protocol; trained (and saved with its per-epoch log) if no checkpoint exists. A local
    weight > 0 adds the local term, built on the elites of the instance whatever the main objective."""
    path = checkpoint_path(arch, objective, instance, seed, local_weight)
    if os.path.exists(path):
        return load_model(arch, path, device)
    partial_path = path + ".partial"
    local = dict(local_dataset=LocalMoveTripletDataset, local_data=training_data("edge", instance),
                 local_weight=local_weight) if local_weight else {}
    model = fit_encoder(training_data(objective, instance), device, dataset=OBJECTIVES[objective], path=partial_path,
                        epochs=PROTOCOL_EPOCHS, fixed_epochs=True, make_model=ARCHITECTURES[arch], seed=seed, **local)
    with open(history_path(path), "w") as f:
        json.dump(model.training_history, f)
    os.replace(partial_path, path)
    return model


def load_history(arch, objective="edge", instance="tsp21", seed=0, local_weight=0.0):
    with open(history_path(checkpoint_path(arch, objective, instance, seed, local_weight))) as f:
        return json.load(f)


def random_init(arch, device, seed=0):
    """Untrained encoder at the initialization fit_encoder starts from with the same seed."""
    torch.manual_seed(TORCH_SEED + seed)
    return ARCHITECTURES[arch]().to(device)


def embed(model, perms, device):
    """Unit embeddings of tours in canonical orientation (the input of every encoder here)."""
    return embed_tours(model, perms, device, canonical=True)


def load_or_generate_runs(path, run_ids, tsp_seed=None):
    if os.path.exists(path):
        return dict(np.load(path))
    data = generate_trajectories(tsp_seed=tsp_seed, run_ids=run_ids)
    np.savez_compressed(path, **data)
    return data


def load_or_generate_test_runs(instance="tsp21"):
    """TEST_RUNS new GA runs on an instance (run ids N_RUNS.., seeds BASE_SEED + run id); never used for training or
    selection."""
    path = TEST_PATH if instance == "tsp21" else INSTANCE_TEST_PATH.format(INSTANCE_SEEDS[instance])
    return load_or_generate_runs(path, np.arange(N_RUNS, N_RUNS + TEST_RUNS), INSTANCE_SEEDS[instance])


def load_or_generate_dev_runs():
    """DEV_RUNS more GA runs on instance 1 (run ids DEV_FIRST_RUN.., seeds outside training and test); the selection
    set of the final iteration."""
    return load_or_generate_runs(DEV_PATH, np.arange(DEV_FIRST_RUN, DEV_FIRST_RUN + DEV_RUNS))


def evaluation_sets():
    val_runs, _ = split_runs()
    instance_2 = instance_elites("tsp99")
    return {"instance 1, validation runs": runs_subset(instance_elites("tsp21"), val_runs),
            "instance 1, test runs": drop_repeated_steps(load_or_generate_test_runs()),
            "instance 2, all runs": instance_2,
            "instance 2, validation runs": runs_subset(instance_2, val_runs)}


def flat_check(history, window=20):
    """Whether the losses flatten at the end: mean train / val loss over the first and the second-to-last and last
    windows of epochs, and the relative change between the last two windows."""
    train = np.array([h["train_loss"] for h in history])
    val = np.array([h["val_loss"] for h in history])
    out = {"epochs": len(history)}
    for name, x in (("train", train), ("val", val)):
        prev, last = x[-2 * window:-window].mean(), x[-window:].mean()
        out[f"{name} first {window}"] = float(x[:window].mean())
        out[f"{name} prev {window}"] = float(prev)
        out[f"{name} last {window}"] = float(last)
        out[f"{name} change"] = float((last - prev) / prev)
    out["val acc last"] = float(np.mean([h["val_acc"] for h in history[-window:]]))
    return out


def soft_agreement(data, Z, k, seed=STN_CENTERS_SEED):
    """Soft agreement of the embedding with the permutation space for k centers (the centers of analyse_k): per tour
    the Spearman correlation between its distances to the k centers in edge distance and in the embedding (ties
    ranked by their mean), averaged over tours."""
    centers = sample_centers(data["perm"], k, seed)
    a = rankdata(edge_distance(data["perm"], data["perm"][centers]), axis=1)
    b = rankdata(embedding_distance(Z, Z[centers]), axis=1)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return float(np.nanmean(r))


def soft_ks(ks):
    return [k for k in ks if k >= 20] or list(ks[-2:])  # small configs have no k >= 20


def evaluate(models, data, device, ks=STN_N_CENTERS):
    """Geometry rows, STN tables (stn_pivots plus the soft agreement for k >= 20) and analyse_k results of the given
    encoders on one evaluation set; models maps a row name to an encoder (inputs in canonical orientation)."""
    Z = {name: embed(model, data["perm"], device) for name, model in models.items()}
    geometry, _ = geometry_rows(Z, data)
    tables, results = stn_pivots(data, Z, np.unique(data["run_id"]), ks)
    tables["soft"] = [{"encoder": name, **{f"k={k}": soft_agreement(data, Zn, k) for k in soft_ks(ks)}}
                      for name, Zn in Z.items()]
    return geometry, tables, results


def selection_score(tables, name, reference=REFERENCE, ks=(20, 50)):
    """Selection criterion of one trained encoder (row name), computed on the validation runs of instance 1: mean
    difference to the reference (untrained LSTM) of ARI and end_equal over k in ks."""
    cols = [f"k={k}" for k in ks]
    diffs = []
    for metric in ("ari", "end_equal"):
        rows = {r["encoder"]: r for r in tables[metric]}
        diffs += [rows[name][c] - rows[reference][c] for c in cols]
    return float(np.mean(diffs))



# simplest and cheapest first: no recurrence and fewest parameters, then a set model, then the LSTM
SIMPLICITY = ("incidence_mlp", "edge_deepsets", "lstm")


def select_architecture(scores, simplicity=SIMPLICITY):
    """The selection rule, fixed before the results of all seeds were seen.

    scores maps an architecture to its selection_score per seed on the validation runs of instance 1 (the test runs
    and instance 2 are only reported, never used for the choice). The architecture with the highest mean score wins,
    unless others are tied with it: an architecture is tied with the top one when the difference of their means is
    smaller than the larger of their two standard deviations over seeds. Among the tied ones the simplest and
    cheapest wins (order simplicity, default SIMPLICITY; for local weights the smaller weight). Returns the choice and
    per architecture the mean, the standard deviation and whether it is tied with the top one.
    """
    stats = {a: (float(np.mean(v)), float(np.std(v, ddof=1))) for a, v in scores.items()}
    top = max(stats, key=lambda a: stats[a][0])
    tied = [a for a in stats if a == top or stats[top][0] - stats[a][0] < max(stats[top][1], stats[a][1])]
    choice = min(tied, key=list(simplicity).index)
    return choice, [{"architecture": a, "mean": m, "sd": sd, "tied with top": a in tied}
                    for a, (m, sd) in stats.items()]

# Row names: trained encoders '<arch>[ <variant>] s<seed>', untrained ones 'random init <arch> s<seed>'.
SEED_NAME = re.compile(r"(.+) s(\d+)")
GEOMETRY_COLUMNS = ["recall@5", "recall@10", "spearman", "run_acc", "gen_r2"]


def is_trained(name):
    return SEED_NAME.fullmatch(name) is not None and not name.startswith("random init")


def own_init(name):
    """Row of the untrained encoder a trained row starts from (same architecture, same seed)."""
    model, seed = SEED_NAME.fullmatch(name).groups()
    return f"random init {model.split()[0]} s{seed}"


def gain_rows(table, columns, reference=REFERENCE, diffs=True):
    """Per row the values of the columns and, for trained rows, the differences to the own random init (gain from
    training) and to the reference (untrained LSTM)."""
    rows = {r["encoder"]: r for r in table}
    out = []
    for name, r in rows.items():
        row = {"encoder": name, "value": {c: float(r[c]) for c in columns}}
        if diffs and is_trained(name):
            row["own"] = {c: float(r[c] - rows[own_init(name)][c]) for c in columns}
            row["lstm"] = {c: float(r[c] - rows[reference][c]) for c in columns}
        out.append(row)
    return out


def format_gains(gains, columns, digits=3):
    """gain_rows with the seeds of a model merged: 'mean±sd (gain over own init) [difference to the untrained LSTM]'
    per column, the differences averaged over seeds; rows without differences as 'mean±sd'."""
    groups = {}
    for g in gains:
        m = SEED_NAME.fullmatch(g["encoder"])
        groups.setdefault(m.group(1) if m else g["encoder"], []).append(g)
    out = []
    for key, gs in groups.items():
        seeded = SEED_NAME.fullmatch(gs[0]["encoder"]) is not None
        row = {"encoder": f"{key} ({len(gs)} seed{'s' if len(gs) > 1 else ''})" if seeded else key}
        for c in columns:
            v = np.array([g["value"][c] for g in gs])
            cell = f"{v.mean():.{digits}f}" + (f"±{v.std(ddof=1):.{digits}f}" if len(v) > 1 else "")
            if "own" in gs[0]:
                cell += (f" ({np.mean([g['own'][c] for g in gs]):+.{digits}f})"
                         f" [{np.mean([g['lstm'][c] for g in gs]):+.{digits}f}]")
            row[c] = cell
        out.append(row)
    return out


def k20_table(tables, metrics=("ari", "soft", "end_equal", "best_strength"), ks=(20, 50)):
    """One row per encoder with the mean over k in ks of each metric (columns '<metric> k>=20')."""
    cols = [f"k={k}" for k in ks]
    rows = {}
    for m in metrics:
        for r in tables[m]:
            rows.setdefault(r["encoder"], {"encoder": r["encoder"]})[f"{m} k>=20"] = float(np.mean([r[c] for c in cols]))
    return [r for r in rows.values() if len(r) == len(metrics) + 1]


def end_equal_chance_table(results, data):
    """Chance level of end_equal (sum over locations of p_perm * p_emb of the final locations) per encoder and k."""
    rows = {}
    for (name, k), res in results.items():
        rows.setdefault(name, {"encoder": name})[f"k={k}"] = end_equal_chance(res["perm_labels"], res["emb_labels"],
                                                                              data["run_id"])
    return list(rows.values())


def gain_report(geometry, tables, results, data, ks=STN_N_CENTERS):
    """The report tables of one evaluation set, every trained row as 'mean±sd (gain over own init) [difference to
    the untrained LSTM]': metrics at k >= 20, geometry and probes, ARI and the soft agreement, best_strength / n_end /
    end_equal per k (the permutation space and the tie-breaking ceiling as absolute rows) and the chance level of
    end_equal."""
    cols = [f"k={k}" for k in ks]
    k20 = k20_table(tables, ks=soft_ks(ks))
    report = {"mean over k = 20, 50": format_gains(gain_rows(k20, list(k20[0])[1:]), list(k20[0])[1:]),
              "geometry and probes": format_gains(gain_rows(geometry, GEOMETRY_COLUMNS), GEOMETRY_COLUMNS)}
    report["ari"] = format_gains(gain_rows(tables["ari"], cols), cols)
    soft_cols = [f"k={k}" for k in soft_ks(ks)]
    report["soft agreement (per-tour Spearman of the distances to the centers)"] = format_gains(
        gain_rows(tables["soft"], soft_cols), soft_cols)
    for metric in ("best_strength", "n_end", "end_equal"):
        report[metric] = format_gains(gain_rows(tables[metric], cols), cols)
    chance = end_equal_chance_table(results, data)
    report["end_equal chance level (sum p_perm * p_emb)"] = format_gains(gain_rows(chance, cols, diffs=False), cols,
                                                                         digits=2)
    return report
