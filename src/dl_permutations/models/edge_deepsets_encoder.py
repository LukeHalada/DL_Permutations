import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeDeepSetsEncoder(nn.Module):
    """DeepSets over the undirected edges of the full cycle.

    The input is a tour without its fixed start cities (0..start-1), as for RNNEncoder; the full cycle is built inside.
    Every edge {a, b} is embedded by phi([e_a + e_b, e_a * e_b]) from learned city embeddings e (symmetric in a and b),
    the edge embeddings are mean-pooled and mapped by rho. The embedding depends only on the set of undirected edges:
    it is invariant to the direction and the rotation of the cycle by construction.
    """

    def __init__(self, n_cities, start=1, emb_dim=32, hidden_size=128, out_dim=64, dropout=0.1, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.register_buffer("start_cities", torch.arange(start))
        self.embedding = nn.Embedding(n_cities, emb_dim)
        self.phi = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_dim),
        )

    def full_cycle(self, x):
        return torch.cat([self.start_cities.expand(len(x), -1), x.long()], dim=1)

    def embed_cycle(self, cycle):
        a, b = self.embedding(cycle), self.embedding(cycle.roll(-1, dims=1))
        edges = self.phi(torch.cat([a + b, a * b], dim=-1))
        z = self.rho(edges.mean(dim=1))
        return F.normalize(z, dim=1) if self.normalize else z

    def forward(self, x):
        return self.embed_cycle(self.full_cycle(x))
