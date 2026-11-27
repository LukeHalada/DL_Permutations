"""Encoders for the STN comparison and the GA runs they are trained and evaluated on.

Three architectures: the pipeline's LSTM (RNNEncoder), an MLP on the edge incidence vector and DeepSets over the
edges. All are trained on triplets of elites ordered by edge distance for a fixed number of epochs (the checkpoint is
the last epoch), on tours in canonical orientation, with the run split and hyperparameters of the pipeline. A missing
checkpoint is trained on first use. Instances: tsp21 (the pipeline's instance) and the replication instances.
"""
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .contrastive_trajectory_learning import (
    BASE_SEED, BATCH_SIZE, CITIES_COUNT, COORD_MAX, COORD_MIN, CROSS_RATE, DATA_HASH, DROPOUT, EMB_DIM, HIDDEN_SIZE, LR,
    MARGIN, MUTATION_RATE, N_GENERATIONS, N_RUNS, OUT_DIM, OUTPUT_DIR, POPULATION_SIZE, START_CITY, TORCH_SEED,
    TOURNAMENT_SIZE, TRIPLETS_PER_EPOCH, TSP_SEED, VAL_RUN_RATIO, WEIGHT_DECAY, config, drop_repeated_steps,
    evaluate_triplets, load_or_generate_trajectories, make_encoder,
)
from .genetic_alg.ga import GA_TSP
from .genetic_alg.tsp import TSP
from .models.edge_deepsets_encoder import EdgeDeepSetsEncoder
from .models.incidence_mlp_encoder import IncidenceMLPEncoder
from .stn_clusters import edge_distance, stn_config, unit_norm

EPOCHS = stn_config['encoder_protocol_config']['epochs']
SEEDS = stn_config['encoder_protocol_config']['seeds']
MIN_GAP = stn_config['edge_triplet_config']['min_gap']
TEST_RUNS = stn_config['test_set_config']['n_runs']
INSTANCES = {"tsp21": TSP_SEED, **{f"tsp{seed}": seed for seed in stn_config['replication_config']['tsp_seeds']}}
EMBED_BATCH = 4096


def config_hash(*sections):
    """Short hash of config sections of the pipeline config and stn_config.yaml, as the pipeline's cache names."""
    merged = {**config, **stn_config}
    return hashlib.sha256(json.dumps({s: merged[s] for s in sections}, sort_keys=True).encode()).hexdigest()[:8]


MODEL_HASH = config_hash('tsp_config', 'ga_config', 'training_config', 'model_config', 'edge_triplet_config',
                         'encoder_protocol_config')
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "encoder_{arch}_edge_{instance}_s{seed}_" + MODEL_HASH + ".pt")


def make_incidence_mlp():
    return IncidenceMLPEncoder(CITIES_COUNT, START_CITY, HIDDEN_SIZE, OUT_DIM, DROPOUT, normalize=True)


def make_edge_deepsets():
    return EdgeDeepSetsEncoder(CITIES_COUNT, START_CITY, EMB_DIM, HIDDEN_SIZE, OUT_DIM, DROPOUT, normalize=True)


ARCHITECTURES = {"lstm": make_encoder, "incidence_mlp": make_incidence_mlp, "edge_deepsets": make_edge_deepsets}


def canonical_orientation(perms):
    """Each tour in the direction with perm[0] < perm[-1] (city 0 fixed at the start), one input per cycle."""
    assert START_CITY == 1, "perm[::-1] is the reversed cycle only with exactly one fixed start city"
    perms = np.atleast_2d(perms)
    return np.where((perms[:, 0] > perms[:, -1])[:, None], perms[:, ::-1], perms)


# --- GA runs

def run_ga(run_id, tsp_seed):
    """The elite of every generation of one GA run (seed BASE_SEED + run_id) on the instance tsp_seed."""
    tsp = TSP(CITIES_COUNT, min=COORD_MIN, max=COORD_MAX, start=START_CITY, seed=tsp_seed)
    np.random.seed(BASE_SEED + run_id)
    ga = GA_TSP(tsp.cities, POPULATION_SIZE, MUTATION_RATE, CROSS_RATE, TOURNAMENT_SIZE, tsp.total_distance)
    history = ga.fit(N_GENERATIONS, tsp.generate_solutions(POPULATION_SIZE))[1]
    return history, np.array([tsp.total_distance(p) for p in history])


def load_or_generate_runs(path, tsp_seed, run_ids):
    if os.path.exists(path):
        return dict(np.load(path))
    runs = [run_ga(run, tsp_seed) for run in run_ids]
    data = {"run_id": np.repeat(run_ids, N_GENERATIONS), "generation": np.tile(np.arange(N_GENERATIONS), len(run_ids)),
            "perm": np.concatenate([h for h, _ in runs]), "fitness": np.concatenate([f for _, f in runs])}
    np.savez_compressed(path, **data)
    return data


def elites(instance):
    """Elites of the training runs 0..N_RUNS-1 of an instance (repeated steps dropped)."""
    if instance == "tsp21":
        raw = load_or_generate_trajectories()
    else:
        path = os.path.join(OUTPUT_DIR, f"trajectories_{instance}_{DATA_HASH}.npz")
        raw = load_or_generate_runs(path, INSTANCES[instance], np.arange(N_RUNS))
    return drop_repeated_steps(raw)


def test_runs(instance):
    """TEST_RUNS held-out GA runs of an instance (run ids N_RUNS.., seeds BASE_SEED + run id), never trained on."""
    suffix = "" if instance == "tsp21" else f"_{instance}"
    path = os.path.join(OUTPUT_DIR, f"trajectories_test{TEST_RUNS}{suffix}_{DATA_HASH}.npz")
    return drop_repeated_steps(load_or_generate_runs(path, INSTANCES[instance], np.arange(N_RUNS, N_RUNS + TEST_RUNS)))


def split_runs():
    """Validation and training runs, as in the pipeline."""
    shuffled = np.random.default_rng(TORCH_SEED).permutation(N_RUNS)
    n_val = int(N_RUNS * VAL_RUN_RATIO)
    return shuffled[:n_val], shuffled[n_val:]


def training_elites(instance):
    """Elites of the training runs of an instance (validation runs left out): what every encoder is trained on."""
    data = elites(instance)
    keep = np.isin(data["run_id"], split_runs()[1])
    return {key: value[keep] for key, value in data.items()}


# --- training

class EdgeTripletDataset(Dataset):
    """Triplets of three different elites of the given runs, drawn uniformly, with the positive closer to the anchor
    than the negative by at least MIN_GAP edges."""

    def __init__(self, data, runs, length, seed):
        self.perm = data["perm"][np.isin(data["run_id"], runs)]
        self.dist = edge_distance(self.perm, self.perm)
        self.length = length
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        while True:
            a, p, n = self.rng.choice(len(self.perm), size=3, replace=False)
            if abs(int(self.dist[a, p]) - int(self.dist[a, n])) >= MIN_GAP:
                break
        a, p, n = (a, p, n) if self.dist[a, p] < self.dist[a, n] else (a, n, p)
        return tuple(torch.as_tensor(self.perm[i]) for i in (a, p, n))


def train(arch, instance, seed, device, path):
    """EPOCHS epochs of triplet training of a fresh encoder (torch seed TORCH_SEED + seed); saves the last epoch."""
    data = elites(instance)
    data = {**data, "perm": canonical_orientation(data["perm"])}
    torch.manual_seed(TORCH_SEED + seed)
    val_runs, train_runs = split_runs()
    train_loader = DataLoader(EdgeTripletDataset(data, train_runs, TRIPLETS_PER_EPOCH, seed=1 + 2 * seed),
                              batch_size=BATCH_SIZE)
    val_ds = EdgeTripletDataset(data, val_runs, int(TRIPLETS_PER_EPOCH * VAL_RUN_RATIO), seed=2)
    val_loader = DataLoader([val_ds[i] for i in range(len(val_ds))], batch_size=BATCH_SIZE)
    model = ARCHITECTURES[arch]().to(device)
    criterion = nn.TripletMarginLoss(margin=MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for a, p, n in train_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            loss = criterion(model(a), model(p), model(n))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        val_loss, val_acc = evaluate_triplets(model, val_loader, device)
        print(f"epoch {epoch:3d} | train {total / len(train_loader):.4f} | val {val_loss:.4f} | val acc {val_acc:.3f}")
    torch.save(model.state_dict(), path)


def load_or_train(arch, device, instance="tsp21", seed=0):
    """The trained encoder; trained first if its checkpoint is missing (saved under a temporary name until done)."""
    path = CHECKPOINT_PATH.format(arch=arch, instance=instance, seed=seed)
    if not os.path.exists(path):
        train(arch, instance, seed, device, path + ".partial")
        os.replace(path + ".partial", path)
    model = ARCHITECTURES[arch]().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def random_init(arch, device, seed=0):
    """The untrained encoder that training with the same seed starts from."""
    torch.manual_seed(TORCH_SEED + seed)
    return ARCHITECTURES[arch]().to(device)


@torch.no_grad()
def embed_tours(model, perms, device):
    """Unit-norm embeddings of tours in canonical orientation."""
    model.eval()
    perms = canonical_orientation(perms)
    Z = np.concatenate([model(torch.as_tensor(perms[i:i + EMBED_BATCH]).to(device)).cpu().numpy()
                        for i in range(0, len(perms), EMBED_BATCH)])
    return unit_norm(Z)
