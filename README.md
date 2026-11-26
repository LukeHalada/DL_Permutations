# DL_Permutations

Visualizing permutation spaces using deep learning techniques.

Papers: https://drive.google.com/drive/folders/12XPfBZJrhvgZ8Cdq9mIjcBwNCj0IWOCO?usp=sharing

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

PyTorch is installed from the CUDA 13.0 build (index pinned in `pyproject.toml`). GPU use needs an NVIDIA driver with CUDA 13.0+.

## Usage

```bash
uv run jupyter lab
```

Then open a notebook from [`notebooks/`](notebooks/).

Contrastive trajectory experiment:

```bash
uv run python -m dl_permutations.contrastive_trajectory_learning
```

The same through `make`: `make sync`, `make lab`, `make train`, `make test`, `make notebooks` (executes all notebooks into `build/notebooks/`), `make clean`.

STN with clustered locations, permutation vs. embedding space: `uv run python -m dl_permutations.stn_clusters` (or `make stn`). Controls (tour orientation, untrained encoder, null model, ties, probes): `notebooks/stn_controls.ipynb`. Encoder variants with other training objectives (edge-distance triplets, canonical orientation): `notebooks/encoder_variants.ipynb`. Encoder architectures (LSTM, MLP on the edge incidence vector, DeepSets over edges) under one fixed-epoch protocol, evaluated on held-out GA runs: `notebooks/encoder_architectures.ipynb`; invariance tests: `make test`. Final selection on a dev set with a run bootstrap, replication over five instances and one table of the whole project: `notebooks/project_summary.ipynb`; reproducibility: `REPRODUCIBILITY.md`.

## Structure

```
src/dl_permutations/   # source package (genetic algorithm, models)
notebooks/             # Jupyter notebooks
outputs/               # generated artifacts (git-ignored, except stn_pca.png)
```
