# DL_Permutations

Visualizing permutation spaces using deep learning techniques.

Papers: https://drive.google.com/drive/folders/12XPfBZJrhvgZ8Cdq9mIjcBwNCj0IWOCO?usp=sharing

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

PyTorch is pulled from the CUDA 13.0 build (`pyproject.toml` pins the index) — an NVIDIA driver supporting CUDA 13.0+ is required for GPU acceleration.

## Usage

```bash
uv run jupyter lab
```

Then open a notebook from [`notebooks/`](notebooks/).

## Structure

```
src/dl_permutations/   # source package (genetic algorithm, models)
notebooks/             # Jupyter notebooks
outputs/               # saved model checkpoints
```
