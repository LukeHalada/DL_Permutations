"""Invariance of the edge-set encoders to the direction and rotation of the cycle.

Run with `python -m tests.test_encoder_invariance` (or `make test`) from the repo root: the code imports
`src.dl_permutations...`, so the root must be on sys.path, which `-m` does and running the file does not.
"""
import numpy as np
import torch

from dl_permutations.encoder_architectures import ARCHITECTURES, random_init
from dl_permutations.stn_clusters import edge_matrix
from dl_permutations.stn_controls import canonical_orientation

N_CITIES, START = 20, 1
INVARIANT = ("incidence_mlp", "edge_deepsets")


def random_tours(m=64, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(np.arange(START, N_CITIES)) for _ in range(m)])


@torch.no_grad()
def embeddings(arch, x):
    model = random_init(arch, torch.device("cpu"))
    model.eval()
    return model, model(torch.as_tensor(x))


def test_reversal_and_rotation_invariance():
    perms = random_tours()
    for arch in INVARIANT:
        model, z = embeddings(arch, perms)
        with torch.no_grad():
            assert torch.allclose(z, model(torch.as_tensor(perms[:, ::-1].copy())), atol=1e-6), arch
            assert torch.allclose(z, model(torch.as_tensor(canonical_orientation(perms))), atol=1e-6), arch
            cycle = model.full_cycle(torch.as_tensor(perms))
            for shift in (1, 7, N_CITIES - 1):
                rotated = cycle.roll(shift, dims=1)
                assert torch.allclose(z, model.embed_cycle(rotated), atol=1e-6), (arch, shift)
                assert torch.allclose(z, model.embed_cycle(rotated.flip(1)), atol=1e-6), (arch, shift)


def test_different_cycles_differ():
    perms = random_tours()
    for arch in INVARIANT:
        _, z = embeddings(arch, perms)
        assert torch.cdist(z, z)[~torch.eye(len(z), dtype=torch.bool)].min() > 1e-4, arch


def test_incidence_vector_matches_edge_matrix():
    perms = random_tours()
    model = random_init("incidence_mlp", torch.device("cpu"))
    x = model.incidence(model.full_cycle(torch.as_tensor(perms))).numpy()
    assert np.array_equal(x, edge_matrix(perms)) and (x.sum(axis=1) == N_CITIES).all()


def test_lstm_is_not_invariant():
    perms = random_tours()
    model, z = embeddings("lstm", perms)
    with torch.no_grad():
        assert not torch.allclose(z, model(torch.as_tensor(perms[:, ::-1].copy())), atol=1e-3)


def test_same_interface():
    perms = torch.as_tensor(random_tours(8))
    for arch in ARCHITECTURES:
        _, z = embeddings(arch, perms)
        assert z.shape == (8, 64) and torch.allclose(z.norm(dim=1), torch.ones(8), atol=1e-5), arch


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}  OK")
