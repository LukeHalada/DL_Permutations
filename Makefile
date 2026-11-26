.PHONY: sync lab train notebooks clean

sync:
	uv sync

lab:
	uv run jupyter lab

train:
	uv run python -m dl_permutations.contrastive_trajectory_learning

notebooks:
	uv run jupyter nbconvert --to notebook --execute --output-dir build/notebooks notebooks/*.ipynb

clean:
	rm -rf build
