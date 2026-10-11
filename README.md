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

The same through `make`: `make sync`, `make lab`, `make train`, `make notebooks` (executes all notebooks into `build/notebooks/`), `make clean`.

## Structure

```
src/dl_permutations/   # source package (genetic algorithm, models)
notebooks/             # Jupyter notebooks
outputs/               # generated artifacts (git-ignored, except stn_pca.png)
```
