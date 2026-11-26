import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import yaml

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

from src.dl_permutations.genetic_alg.ga import GA_TSP
from src.dl_permutations.genetic_alg.tsp import TSP
from src.dl_permutations.models.rnn_encoder import RNNEncoder

config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "contrastive_trajectory_config.yaml")))
tsp_config = config['tsp_config']
ga_config = config['ga_config']
model_config = config['model_config']
train_config = config['training_config']

CITIES_COUNT = tsp_config['cities_count']
COORD_MIN, COORD_MAX = tsp_config['min'], tsp_config['max']
START_CITY = tsp_config['start']
TSP_SEED = tsp_config['seed']

N_RUNS = ga_config['n_runs']
N_GENERATIONS = ga_config['n_generations']
POPULATION_SIZE = ga_config['population_size']
MUTATION_RATE = ga_config['mutation_rate']
CROSS_RATE = ga_config['cross_rate']
TOURNAMENT_SIZE = ga_config['tournament_size']
BASE_SEED = ga_config['base_seed']
N_WORKERS = os.cpu_count()

HARD_NEGATIVE_WINDOW = train_config['negative_window']
TRIPLETS_PER_EPOCH = train_config['triplets_per_epoch']
VAL_RUN_RATIO = train_config['val_ratio']
BATCH_SIZE = train_config['batch_size']
EPOCHS = train_config['epochs']
LR = train_config['lr']
WEIGHT_DECAY = train_config['weight_decay']
MAX_PATIENCE = train_config['max_patience']
TORCH_SEED = train_config['torch_seed']


EMB_DIM = model_config['emb_dim']
HIDDEN_SIZE = model_config['hidden_size']
OUT_DIM = model_config['out_dim']
DROPOUT = model_config['dropout']
MARGIN = model_config['margin']

PLOT_RUNS = 3
VARIANCE_THRESHOLD = 0.95

def config_hash(*sections):
    """Short hash of the given config sections. Caches derived from them carry it in their file names, so a changed
    config never silently reuses data computed with the old one (code changes are not covered)."""
    payload = json.dumps({s: config[s] for s in sections}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


DATA_HASH = config_hash('tsp_config', 'ga_config')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
TRAJECTORIES_PATH = os.path.join(OUTPUT_DIR, f"trajectories_{DATA_HASH}.npz")
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "embeddings.npz")
MODEL_PATH = os.path.join(OUTPUT_DIR, "contrastive_encoder.pt")
PLOT_PATH = os.path.join(OUTPUT_DIR, "stn_pca.png")


def make_tsp(seed=None):
    """The TSP instance; seed (default TSP_SEED) draws the city coordinates."""
    return TSP(CITIES_COUNT, min=COORD_MIN, max=COORD_MAX, start=START_CITY,
               seed=TSP_SEED if seed is None else seed)


def run_single_ga(run_id, return_populations=False, tsp_seed=None):
    tsp = make_tsp(tsp_seed)
    np.random.seed(BASE_SEED + run_id)
    ga = GA_TSP(tsp.cities, POPULATION_SIZE, MUTATION_RATE, CROSS_RATE,
                TOURNAMENT_SIZE, tsp.total_distance)
    out = ga.fit(N_GENERATIONS, tsp.generate_solutions(POPULATION_SIZE),
                 return_populations=return_populations)
    return out[1:] if return_populations else out[1]


def generate_trajectories(tsp_seed=None, run_ids=None):
    """Elite of every generation of N_RUNS GA runs; tsp_seed selects another TSP instance (default TSP_SEED), run_ids
    other runs (default 0..N_RUNS-1; run r uses the seed BASE_SEED + r)."""
    tsp = make_tsp(tsp_seed)
    run_ids = np.arange(N_RUNS) if run_ids is None else np.asarray(run_ids)
    with ProcessPoolExecutor(N_WORKERS) as pool:
        histories = list(pool.map(partial(run_single_ga, tsp_seed=tsp_seed), run_ids.tolist()))
    perms = np.concatenate(histories)
    return {
        "run_id": np.repeat(run_ids, N_GENERATIONS),
        "generation": np.tile(np.arange(N_GENERATIONS), len(run_ids)),
        "perm": perms,
        "fitness": np.array([tsp.total_distance(p) for p in perms]),
    }


def load_or_generate_trajectories():
    if os.path.exists(TRAJECTORIES_PATH):
        return dict(np.load(TRAJECTORIES_PATH))
    data = generate_trajectories()
    np.savez_compressed(TRAJECTORIES_PATH, **data)
    return data


def drop_repeated_steps(data):
    keep = np.ones(len(data["run_id"]), dtype=bool)
    same_run = data["run_id"][1:] == data["run_id"][:-1]
    same_perm = (data["perm"][1:] == data["perm"][:-1]).all(axis=1)
    keep[1:] = ~(same_run & same_perm)
    return {k: v[keep] for k, v in data.items()}


class TripletDataset(Dataset):
    def __init__(self, data, runs, length, seed):
        mask = np.isin(data["run_id"], runs)
        self.perm = data["perm"][mask]
        self.fitness = data["fitness"][mask]
        self.run_id = data["run_id"][mask]
        self.length = length
        self.rng = np.random.default_rng(seed)

        has_next = np.zeros(len(self.run_id), dtype=bool)
        has_next[:-1] = self.run_id[1:] == self.run_id[:-1]
        self.anchor_idx = np.flatnonzero(has_next)

        self.order = np.argsort(self.fitness)
        self.rank = np.empty_like(self.order)
        self.rank[self.order] = np.arange(len(self.order))

    def __len__(self):
        return self.length

    def same_cycle(self, i, j):
        a, b = self.perm[i], self.perm[j]
        return np.array_equal(a, b) or np.array_equal(a, b[::-1])

    def mine_negative(self, anchor):
        lo = max(0, self.rank[anchor] - HARD_NEGATIVE_WINDOW)
        hi = min(len(self.order), self.rank[anchor] + HARD_NEGATIVE_WINDOW + 1)
        while True:
            cand = self.order[self.rng.integers(lo, hi)]
            if self.run_id[cand] != self.run_id[anchor] and not self.same_cycle(anchor, cand):
                return cand

    def __getitem__(self, _):
        a = self.anchor_idx[self.rng.integers(len(self.anchor_idx))]
        n = self.mine_negative(a)
        return (torch.as_tensor(self.perm[a]), torch.as_tensor(self.perm[a + 1]),
                torch.as_tensor(self.perm[n]))

def make_encoder():
    return RNNEncoder(CITIES_COUNT, EMB_DIM, HIDDEN_SIZE, OUT_DIM, DROPOUT, normalize=True)


@torch.no_grad()
def evaluate_triplets(model, loader, device):
    model.eval()
    criterion = nn.TripletMarginLoss(margin=MARGIN)
    loss, correct, total = 0.0, 0, 0
    for a, p, n in loader:
        za, zp, zn = (model(t.to(device)) for t in (a, p, n))
        loss += criterion(za, zp, zn).item()
        correct += (F.pairwise_distance(za, zp) < F.pairwise_distance(za, zn)).sum().item()
        total += len(a)
    return loss / len(loader), correct / total


def train_epoch(model, loader, criterion, optimizer, device, local_loader=None, local_weight=0.0):
    """One epoch of triplet training; with local_loader every batch adds local_weight times the triplet loss of a
    batch of local triplets."""
    model.train()
    total = 0.0
    batches = zip(loader, local_loader) if local_loader is not None else ((batch, None) for batch in loader)
    for (a, p, n), local in batches:
        a, p, n = a.to(device), p.to(device), n.to(device)
        loss = criterion(model(a), model(p), model(n))
        if local is not None:
            la, lp, ln = (t.to(device) for t in local)
            loss = loss + local_weight * criterion(model(la), model(lp), model(ln))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def train(model, train_loader, val_loader, device, path=None, epochs=None):
    """Triplet training with early stopping; keeps the checkpoint with the lowest validation loss (path, default
    MODEL_PATH). The protocol of the original pipeline."""
    path = path or MODEL_PATH
    criterion = nn.TripletMarginLoss(margin=MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best, patience = np.inf, 0

    for epoch in range(epochs or EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_triplets(model, val_loader, device)
        print(f"epoch {epoch:2d} | train {train_loss:.4f} | val {val_loss:.4f} | val triplet acc {val_acc:.3f}")

        if val_loss < best:
            best, patience = val_loss, 0
            torch.save(model.state_dict(), path)
        else:
            patience += 1
            if patience >= MAX_PATIENCE:
                break

    model.load_state_dict(torch.load(path, map_location=device))
    return model


def train_fixed_epochs(model, train_loader, val_loader, device, path, epochs, local_loaders=None, local_weight=0.0):
    """Triplet training for exactly `epochs` epochs; the checkpoint (path) is the model after the last epoch.

    The validation loss is only logged for monitoring and selects nothing. The per-epoch log is kept in
    model.training_history. local_loaders (train, validation) add a local term: the training loss of every batch
    becomes loss + local_weight * local loss, and the local validation loss is logged separately.
    """
    criterion = nn.TripletMarginLoss(margin=MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    local_train, local_val = local_loaders or (None, None)
    history = []
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, local_train, local_weight)
        val_loss, val_acc = evaluate_triplets(model, val_loader, device)
        entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
        line = f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | val triplet acc {val_acc:.3f}"
        if local_val is not None:
            entry["val_local_loss"], entry["val_local_acc"] = evaluate_triplets(model, local_val, device)
            line += f" | local val {entry['val_local_loss']:.4f} acc {entry['val_local_acc']:.3f}"
        history.append(entry)
        print(line)
    torch.save(model.state_dict(), path)
    model.training_history = history
    return model


@torch.no_grad()
def embed(model, perms, device):
    model.eval()
    return model(torch.as_tensor(perms).to(device)).cpu().numpy()


def collapse_report(emb, fitness):
    pca = PCA().fit(emb)
    cum = np.cumsum(pca.explained_variance_ratio_)
    dims = int(np.searchsorted(cum, VARIANCE_THRESHOLD) + 1)
    corr = np.corrcoef(pca.transform(emb)[:, 0], fitness)[0, 1]
    print(f"PCA dims for {VARIANCE_THRESHOLD:.0%} variance: {dims}/{emb.shape[1]} | "
          f"PC1 explains {pca.explained_variance_ratio_[0]:.1%} | "
          f"|corr(PC1, tour length)| = {abs(corr):.2f}")


def plot_stn(model, data, device, runs):
    emb_all = embed(model, data["perm"], device)
    pca = PCA(n_components=2).fit(emb_all)

    fig, ax = plt.subplots(figsize=(9, 7))
    for run in runs:
        m = data["run_id"] == run
        xy = pca.transform(emb_all[m])
        line, = ax.plot(xy[:, 0], xy[:, 1], lw=0.8, alpha=0.7, label=f"run {run}")
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=data["generation"][m], cmap="viridis",
                        s=18, edgecolors=line.get_color(), linewidths=0.8)
        ax.scatter(*xy[0], marker="*", s=200, color=line.get_color(), edgecolors="k", zorder=3)
    fig.colorbar(sc, label="generation")
    ax.set_title("Search trajectories in embedding space (PCA)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend()
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.show()


def split_runs():
    shuffled = np.random.default_rng(TORCH_SEED).permutation(N_RUNS)
    n_val = int(N_RUNS * VAL_RUN_RATIO)
    return shuffled[:n_val], shuffled[n_val:]


def fit_encoder(data, device, dataset=TripletDataset, path=None, epochs=None, fixed_epochs=False,
                make_model=make_encoder, seed=0, local_dataset=None, local_data=None, local_weight=0.0):
    """Trains an encoder on triplets from dataset (train runs), validated on triplets of the validation runs.

    The defaults are the original pipeline (RNNEncoder, early stopping, checkpoint MODEL_PATH). Variants reuse it with
    another dataset, model factory, checkpoint path and number of epochs; fixed_epochs=True trains exactly `epochs`
    epochs and keeps the last one (train_fixed_epochs). seed shifts the torch seed (initialization) and the seed of
    the training triplets; seed 0 is the original pipeline, and the validation triplets are the same for every seed.
    local_weight > 0 (fixed_epochs only) adds the local term: triplets of local_dataset on local_data (default data),
    as many per epoch as the main ones, weighted by local_weight; the main triplets and the initialization stay
    those of local_weight = 0.
    """
    torch.manual_seed(TORCH_SEED + seed)
    val_runs, train_runs = split_runs()

    train_ds = dataset(data, train_runs, TRIPLETS_PER_EPOCH, seed=1 + 2 * seed)
    val_ds = dataset(data, val_runs, int(TRIPLETS_PER_EPOCH * VAL_RUN_RATIO), seed=2)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
    val_loader = DataLoader([val_ds[i] for i in range(len(val_ds))], batch_size=BATCH_SIZE)

    local_loaders = None
    if local_weight:
        assert fixed_epochs, "the local term is part of the fixed-epoch protocol only"
        local_data = data if local_data is None else local_data
        local_train = local_dataset(local_data, train_runs, TRIPLETS_PER_EPOCH, seed=101 + 2 * seed)
        local_val = local_dataset(local_data, val_runs, int(TRIPLETS_PER_EPOCH * VAL_RUN_RATIO), seed=102)
        local_loaders = (DataLoader(local_train, batch_size=BATCH_SIZE),
                         DataLoader([local_val[i] for i in range(len(local_val))], batch_size=BATCH_SIZE))

    model = make_model().to(device)
    if fixed_epochs:
        return train_fixed_epochs(model, train_loader, val_loader, device, path, epochs, local_loaders, local_weight)
    return train(model, train_loader, val_loader, device, path, epochs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = drop_repeated_steps(load_or_generate_trajectories())
    val_runs, _ = split_runs()

    model = fit_encoder(data, device)

    collapse_report(embed(model, data["perm"], device), data["fitness"])
    plot_stn(model, data, device, val_runs[:PLOT_RUNS])

if __name__ == "__main__":
    main()
