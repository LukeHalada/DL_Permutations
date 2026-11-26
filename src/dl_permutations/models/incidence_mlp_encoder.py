import torch
import torch.nn as nn
import torch.nn.functional as F


class IncidenceMLPEncoder(nn.Module):
    """MLP on the edge incidence vector of the full cycle.

    The input is a tour without its fixed start cities (0..start-1), as for RNNEncoder; the full cycle is built inside.
    The features are the upper triangle of its symmetric n x n adjacency matrix (n * (n - 1) / 2 binary entries, n of
    them set), so the embedding depends only on the set of undirected edges: it is invariant to the direction and the
    rotation of the cycle by construction.
    """

    def __init__(self, n_cities, start=1, hidden_size=128, out_dim=64, dropout=0.1, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.register_buffer("start_cities", torch.arange(start))
        rows, cols = torch.triu_indices(n_cities, n_cities, offset=1)
        edge_index = torch.full((n_cities, n_cities), -1, dtype=torch.long)
        edge_index[rows, cols] = edge_index[cols, rows] = torch.arange(len(rows))
        self.register_buffer("edge_index", edge_index)
        self.n_features = len(rows)
        self.mlp = nn.Sequential(
            nn.Linear(self.n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_dim),
        )

    def full_cycle(self, x):
        return torch.cat([self.start_cities.expand(len(x), -1), x.long()], dim=1)

    def incidence(self, cycle):
        """Edge incidence vectors (batch, n * (n - 1) / 2) of full cycles (batch, n)."""
        edges = self.edge_index[cycle, cycle.roll(-1, dims=1)]
        return torch.zeros(len(cycle), self.n_features, device=cycle.device).scatter_(1, edges, 1.0)

    def embed_cycle(self, cycle):
        z = self.mlp(self.incidence(cycle))
        return F.normalize(z, dim=1) if self.normalize else z

    def forward(self, x):
        return self.embed_cycle(self.full_cycle(x))
