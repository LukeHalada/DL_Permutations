.PHONY: sync lab train stn test notebooks clean

sync:
	uv sync

lab:
	uv run jupyter lab

train:
	uv run python -m dl_permutations.contrastive_trajectory_learning

stn:
	uv run python -m dl_permutations.stn_clusters

test:
	uv run python -m tests.test_encoder_invariance

notebooks:
	uv run jupyter nbconvert --to notebook --execute --output-dir build/notebooks notebooks/*.ipynb

clean:
	rm -rf build
