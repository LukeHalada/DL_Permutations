import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

from dl_permutations.genetic_alg.ga import GA_TSP
from dl_permutations.genetic_alg.tsp import TSP

CITIES_COUNT = 20
COORD_MIN, COORD_MAX = 0, 10
START_CITY = 1
TSP_SEED = 21

N_RUNS = 100
N_GENERATIONS = 100
POPULATION_SIZE = 50
MUTATION_RATE = 0.1
CROSS_RATE = 0.1
TOURNAMENT_SIZE = 5
BASE_SEED = 1000
N_WORKERS = os.cpu_count()

HARD_NEGATIVE_WINDOW = 20
TRIPLETS_PER_EPOCH = 50_000
VAL_RUN_RATIO = 0.2

EMB_DIM = 32
HIDDEN_SIZE = 128
OUT_DIM = 64
DROPOUT = 0.1
MARGIN = 0.3
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-5
MAX_PATIENCE = 6
TORCH_SEED = 0

PLOT_RUNS = 3
VARIANCE_THRESHOLD = 0.95

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
TRAJECTORIES_PATH = os.path.join(OUTPUT_DIR, "trajectories.npz")
MODEL_PATH = os.path.join(OUTPUT_DIR, "contrastive_encoder.pt")
PLOT_PATH = os.path.join(OUTPUT_DIR, "stn_pca.png")


def make_tsp():
    return TSP(CITIES_COUNT, min=COORD_MIN, max=COORD_MAX, start=START_CITY, seed=TSP_SEED)


def run_single_ga(run_id):
    tsp = make_tsp()
    np.random.seed(BASE_SEED + run_id)
    ga = GA_TSP(tsp.cities, POPULATION_SIZE, MUTATION_RATE, CROSS_RATE,
                TOURNAMENT_SIZE, tsp.total_distance)
    _, history = ga.fit(N_GENERATIONS, tsp.generate_solutions(POPULATION_SIZE))
    return history


def generate_trajectories():
    tsp = make_tsp()
    with ProcessPoolExecutor(N_WORKERS) as pool:
        histories = list(pool.map(run_single_ga, range(N_RUNS)))
    perms = np.concatenate(histories)
    return {
        "run_id": np.repeat(np.arange(N_RUNS), N_GENERATIONS),
        "generation": np.tile(np.arange(N_GENERATIONS), N_RUNS),
        "perm": perms,
        "fitness": np.array([tsp.total_distance(p) for p in perms]),
    }


def load_or_generate_trajectories():
    if os.path.exists(TRAJECTORIES_PATH):
        return dict(np.load(TRAJECTORIES_PATH))
    data = generate_trajectories()
    np.savez(TRAJECTORIES_PATH, **data)
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


class RNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(CITIES_COUNT, EMB_DIM)
        self.lstm = nn.LSTM(EMB_DIM, HIDDEN_SIZE, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, OUT_DIM),
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(self.embedding(x))
        return F.normalize(self.proj(hn.squeeze(0)), dim=1)


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


def train(model, train_loader, val_loader, device):
    criterion = nn.TripletMarginLoss(margin=MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best, patience = np.inf, 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for a, p, n in train_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            loss = criterion(model(a), model(p), model(n))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_acc = evaluate_triplets(model, val_loader, device)
        print(f"epoch {epoch:2d} | train {train_loss / len(train_loader):.4f} | "
              f"val {val_loss:.4f} | val triplet acc {val_acc:.3f}")

        if val_loss < best:
            best, patience = val_loss, 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience += 1
            if patience >= MAX_PATIENCE:
                break

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
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


def main():
    torch.manual_seed(TORCH_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = drop_repeated_steps(load_or_generate_trajectories())

    shuffled = np.random.default_rng(TORCH_SEED).permutation(N_RUNS)
    n_val = int(N_RUNS * VAL_RUN_RATIO)
    val_runs, train_runs = shuffled[:n_val], shuffled[n_val:]

    train_ds = TripletDataset(data, train_runs, TRIPLETS_PER_EPOCH, seed=1)
    val_ds = TripletDataset(data, val_runs, int(TRIPLETS_PER_EPOCH * VAL_RUN_RATIO), seed=2)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
    val_loader = DataLoader([val_ds[i] for i in range(len(val_ds))], batch_size=BATCH_SIZE)

    model = train(RNNEncoder().to(device), train_loader, val_loader, device)

    collapse_report(embed(model, data["perm"], device), data["fitness"])
    plot_stn(model, data, device, val_runs[:PLOT_RUNS])


if __name__ == "__main__":
    main()
