# Reproducibility

Final iteration of the STN comparison (permutation space vs. encoder embedding): `notebooks/encoder_architectures.ipynb`
and `notebooks/project_summary.ipynb`. The numbers in `notebooks/project_summary.ipynb`, `outputs/project_table.csv`
and `outputs/final_*.png` were computed with the code of commit `5d870d2` (branch `stn-clusters`; the notebook was
executed there, commit `5d870d2`); `notebooks/encoder_architectures.ipynb` keeps its outputs from commit
`87b6409`. The seeds, the k values, the variants and the selection rules were committed before the final results
were computed (commit `bacf6dc`); the bootstrap test of the absolute metrics, the final figures and the baselines
without learning (PCA of the incidence vectors, classical MDS of the edge distances; no randomness) were added after the
results, at the reviewer's request. v1 (tag `v1-wyniki`, commit `a0aadd6`) is reported as history only.

## Environment

| component | version |
|---|---|
| python | 3.12.10 |
| platform | Windows-11-10.0.26200-SP0 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| torch | 2.14.0+cu130 |
| networkx | 3.6.1 |
| matplotlib | 3.11.2 |
| pyyaml | 6.0.3 |
| nbformat | 5.11.1 |
| nbconvert | 7.17.1 |
| cuda (torch build) | 13.0 |
| gpu | NVIDIA GeForce RTX 3050 Laptop GPU |

Windows 11, run from Windows (not WSL); the environment lives outside the repository (`UV_PROJECT_ENVIRONMENT`)
because the repository's `.venv` belongs to WSL.

## Seeds

All values come from `src/dl_permutations/contrastive_trajectory_config.yaml`; file names carry hashes of the relevant
config sections (data `d72f6534`, encoder protocol `c5cddca1`, local term `df1dfd0d`), so a changed
config writes new files instead of reusing old ones.

| what | seeds |
|---|---|
| TSP instances (20 cities, coordinates in [0, 10], start city 1) | instance 1: 21 (`tsp_config.seed`); instance 2 (transfer): 99; replication: 101, 102, 103, 104, 105 |
| GA run with run id r | `np.random.seed(1000 + r)` (`ga_config.base_seed` + run id), the same on every instance |
| training runs | run ids 0-99 (seeds 1000-1099); validation = 20 of them, held out from the training triplets: `default_rng(0).permutation(100)[:20]` (`split_runs`) |
| test runs | run ids 100-199 (seeds 1100-1199), every instance; reporting only |
| dev runs | run ids 200-299 (seeds 1200-1299), instance 1 only; every final selection |
| populations | the training runs 0-99 re-run with every population kept (same seeds) |
| encoder initialization | `torch.manual_seed(0 + seed)` (`training_config.torch_seed` + seed); seeds 0, 1, 2 on instance 1, seed 0 on the other instances |
| triplets | training `1 + 2 * seed`, validation 2; local term: training `101 + 2 * seed`, validation 102 |
| STN centers | `stn_config.centers_seed` 0, k in {5, 10, 20, 50}; selection criterion on k in {20, 50} |
| bootstrap over runs | seed 0, 1000 resamples (`selection_config`) |
| generation probe | ridge with GroupKFold by run (no randomness) |

## Commands

From the repository root (the code imports `src.dl_permutations...`, so the root must be the working directory):

```
uv sync
make test
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/encoder_architectures.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/project_summary.ipynb
```

(On Windows `make` is `mingw32-make`.) The notebooks create every missing GA data file and train every missing
checkpoint on first use (`load_or_generate_*`, `load_or_fit`); serially that takes long (one incidence_mlp checkpoint:
200 epochs, about 18-30 min on the GPU below with several trainings at once). Here the checkpoints were trained in
parallel beforehand, one process per checkpoint while enough memory was free, with the call the notebooks make:

```
uv run python -c "import torch, dl_permutations.encoder_architectures as A, dl_permutations.contrastive_trajectory_learning as C, dl_permutations.stn_clusters as S; C.N_WORKERS = S.N_WORKERS = 3; A.load_or_fit('incidence_mlp', torch.device('cuda'), objective='edge', instance='tsp101', seed=0, local_weight=0.0)"
```

with arch in {lstm, incidence_mlp, edge_deepsets}, objective in {edge, population_edge, elite_synthetic}, instance in
{tsp21, tsp99, tsp101, ..., tsp105} and local_weight in {0, 0.25, 0.5, 1}; the checkpoints used are the ones listed
below. Generate the data first in one process (`A.instance_elites(instance)`, `A.population_tours(instance)`,
`A.load_or_generate_test_runs(instance)`, `A.load_or_generate_dev_runs()`) so that parallel trainings do not write the
same file. `N_WORKERS = 3`: every GA worker process imports torch with its CUDA libraries, and more workers exceed the
Windows commit limit; the GA results do not depend on the number of workers (every run has its own seed). A
checkpoint gets its final name only after the last epoch (`.partial` before), so an interrupted training is never
loaded as finished.

## Determinism

- GA data are deterministic: stored runs were compared with fresh single-process runs of the same seed (training run 7,
  test run 150 and dev run 250 of instance 1, training run 42 of tsp103, test run 199 of tsp105): identical.
- Training runs on the GPU without `torch.use_deterministic_algorithms`, so a retrained checkpoint can differ in the
  last digits. The hashes below identify the checkpoints behind the reported numbers; the evaluation of a given
  checkpoint has no randomness other than the seeded centers and bootstrap (the probe uses GroupKFold).
- `outputs/stn_pca.png` (original pipeline) was not changed in this iteration.
- `outputs/` is not tracked by git; the executed notebooks are.

## Output files

### Results

| file | bytes | sha256 |
|---|---:|---|
| `outputs/project_table.csv` | 10730 | `756c1482be8f84c215b69ce25ec0800ad0e0ebaeab7e9fdefba4ee0f5c661303` |

### Final figures (incidence_mlp s0, the PCA(2) baseline and the MDS(64) reference, test runs of instance 1)

| file | bytes | sha256 |
|---|---:|---|
| `outputs/final_stn_incidence_mlp_k10.png` | 470910 | `2f28443e2a25e69fc076b193006ffeaba63a81610ac63fa1331638621fa33d81` |
| `outputs/final_stn_incidence_mlp_k20.png` | 718885 | `36e6de202060385e5540fcd30e6041ec48db39928a268142947ae8d4c59b52cf` |
| `outputs/final_trajectories_pca_incidence_mlp.png` | 207853 | `0dd0dc1e5b766b691759dd238e265788fce2ca67bd42ecd637e778493d9b3396` |
| `outputs/final_trajectories_pca_incidence_vs_mlp.png` | 406388 | `a152f80902e5ef217ba67be418d6c84a44a32b343f2803c08982d5c97525d4fd` |
| `outputs/final_stn_perm_mds_mlp_k10.png` | 673085 | `0249b819490e7c737d691f06e81f715791f58ee7b06778e2dc3593f929404de3` |
| `outputs/final_stn_perm_mds_mlp_k20.png` | 1029688 | `c9971f0e89fa8b5963b68a8d0aea14d07226a2875cc3d96cc79b6098bd6143eb` |

### GA data (trajectories of elites, test and dev runs, populations)

| file | bytes | sha256 |
|---|---:|---|
| `outputs/populations_d72f6534.npz` | 5879465 | `f947cd91b2744e07fbee66db36109da3b1f4f63d9e72fddae2a35ed56dc39e1d` |
| `outputs/populations_tsp101_d72f6534.npz` | 5897943 | `3edaa18446b79c4202238a03be8987d6fb700091616b8c0136d8e6145775a374` |
| `outputs/populations_tsp102_d72f6534.npz` | 5881022 | `0355a3e694a0d05874ee8f0b46dbe175877ceba6ed5e0332ddf3716daf9cbe5f` |
| `outputs/populations_tsp103_d72f6534.npz` | 5897451 | `8c94aada8cbbf34221aecbfb7c4d623ce0d7aa89b4eab4391af6def8d5561f22` |
| `outputs/populations_tsp104_d72f6534.npz` | 5893701 | `53f6db115c8725ca14003b19d22065cdd4314cdd12d245496b0b1afc7da588e2` |
| `outputs/populations_tsp105_d72f6534.npz` | 5844579 | `29b80bca14b5819b726d62ca75969661f3e51dbc5473a591c6dca670968ed660` |
| `outputs/trajectories_d72f6534.npz` | 45614 | `f31042ec90744a98e20616ff7d652534040893da3d3c7063ca9b7c45af0ee03f` |
| `outputs/trajectories_dev100_r200_d72f6534.npz` | 45011 | `1b2e9fdede5de182c94cac9fcd440d0294e31f9d54dc582f26525ed82cade0a5` |
| `outputs/trajectories_test100_d72f6534.npz` | 44216 | `5fc79d7125a4d8680263eec96524e4770dfca5654505a98f479e8eb9d4e89603` |
| `outputs/trajectories_test100_tsp101_d72f6534.npz` | 43973 | `11e6c4e73bc0e504f9112e3e1daf1d3ea0d7ae9b8082c8af4d5cc36f2695879b` |
| `outputs/trajectories_test100_tsp102_d72f6534.npz` | 45463 | `cd5c0be2172fde3bb17c7422fae69157dd3e4e3f3453a326573ed5bbb5afdbe8` |
| `outputs/trajectories_test100_tsp103_d72f6534.npz` | 47023 | `5e8d3bcf4f4dab7ace00ad0419260bc5ea778bd2b8b9b857a4281f7a524aa668` |
| `outputs/trajectories_test100_tsp104_d72f6534.npz` | 46402 | `fb74383c49d395c7251e9d707ce4f7ab56dff3bbce57dbf9f8a5c665c0db4525` |
| `outputs/trajectories_test100_tsp105_d72f6534.npz` | 46489 | `f8294e830a2d126cfa3ef48248532e163e3cc5960b2709754a45f4b6d3bc3dbc` |
| `outputs/trajectories_tsp101_d72f6534.npz` | 43688 | `542e5e47fe18cb54b34b64bb9a4e42c439f454b13434848148a68540c37a109d` |
| `outputs/trajectories_tsp102_d72f6534.npz` | 45031 | `37ee3d1fba0882be0002fb58c9aa10c15d41c0cc46e4c6d6158e5fa5c6b5ce32` |
| `outputs/trajectories_tsp103_d72f6534.npz` | 47073 | `c026e1422dab2471a78792289add479eb3b0f313533d5eac4d32658663bffce9` |
| `outputs/trajectories_tsp104_d72f6534.npz` | 46510 | `b4ef86f427bbc9c56cdddb8732657f49407bf15298e4a6a552fa981caaf3ffb5` |
| `outputs/trajectories_tsp105_d72f6534.npz` | 46765 | `ab699bd2e723a6771a9e80b8608aa73e3cd7aea313e6fbcbf7705bc1f8afe431` |
| `outputs/trajectories_tsp99_d72f6534.npz` | 45983 | `d974ab49a5d5705f2832c30dd9e51dba6372333db1378f5a57652ce3db371688` |

### Encoder checkpoints (protocol: 200 epochs, last epoch)

| file | bytes | sha256 |
|---|---:|---|
| `outputs/encoder_edge_deepsets_edge_tsp21_s0_c5cddca1.pt` | 206021 | `eeaaa306b65f7c878370e7efce32a52cb578635a51822d558c918daa66be5ba1` |
| `outputs/encoder_edge_deepsets_edge_tsp21_s1_c5cddca1.pt` | 206021 | `9f5296607973d18fe3582ebbf5662a85f01b3adcde768c63635f1cf68b6df8f4` |
| `outputs/encoder_edge_deepsets_edge_tsp21_s2_c5cddca1.pt` | 206021 | `b022d0937db55157a1f39826fe4e9a31053ae462cc2b6542d0258ef27080df62` |
| `outputs/encoder_edge_deepsets_edge_tsp99_s0_c5cddca1.pt` | 206021 | `3e336a1b6cd6f277b0c0fbcf065c0d5135b072caf7cbd46ce509c4ae3a437f23` |
| `outputs/encoder_edge_deepsets_elite_synthetic_tsp21_s0_c5cddca1.pt` | 206261 | `004fa7fe35f9f960339f0a72050d017167ed6f8ae19a9b8d56168d3a74fde1ff` |
| `outputs/encoder_edge_deepsets_population_edge_tsp21_s0_c5cddca1.pt` | 206261 | `860895e1849e6199dc81fbf95894e483c4cd5eb5577b33ad075aa39166291585` |
| `outputs/encoder_incidence_mlp_edge+local0.25-df1dfd0d_tsp21_s0_c5cddca1.pt` | 204999 | `bbb6c66cdcb8ecde16775451e1e00c8503b19537faa016cf63a7dfa9225de20c` |
| `outputs/encoder_incidence_mlp_edge+local0.5-df1dfd0d_tsp21_s0_c5cddca1.pt` | 204921 | `b7b6a7ad5f6edaa6cb608ccdefc4c68ab09fb61aaee910bcc1133c767c9dd5f5` |
| `outputs/encoder_incidence_mlp_edge+local1-df1dfd0d_tsp21_s0_c5cddca1.pt` | 204765 | `61cfc1be86c25f4ec9a2592ec62a92cf3c9617e8febaca2c0cd0e2c5dfa1518f` |
| `outputs/encoder_incidence_mlp_edge+local1-df1dfd0d_tsp21_s1_c5cddca1.pt` | 204765 | `9525105bbe7fe1cfdb52d43fa1fec44aa678a55fabc9bd0282a93017c2a972fd` |
| `outputs/encoder_incidence_mlp_edge+local1-df1dfd0d_tsp21_s2_c5cddca1.pt` | 204765 | `ce64f40157e9b9e6fbbbd111f6cd9cd7f88cd0273dba5374a7a7bfe5daaaf622` |
| `outputs/encoder_incidence_mlp_edge_tsp101_s0_c5cddca1.pt` | 204427 | `a0cbc6c32956b037895749de3e871e154c8b08f9bdc6425aa957819b68a8a19f` |
| `outputs/encoder_incidence_mlp_edge_tsp102_s0_c5cddca1.pt` | 204427 | `59450d0c8cb7509459de5ac48558c0b8af6c46ff752a471c889bb13efc92f159` |
| `outputs/encoder_incidence_mlp_edge_tsp103_s0_c5cddca1.pt` | 204427 | `4a258fb7a9b1042c6035919961d8f6d201be3dcab2037e15787e2ab0d1c92629` |
| `outputs/encoder_incidence_mlp_edge_tsp104_s0_c5cddca1.pt` | 204427 | `25fbbe3fca821362791c85fe82b71e265241c947568feacfe7cefedd500a9463` |
| `outputs/encoder_incidence_mlp_edge_tsp105_s0_c5cddca1.pt` | 204427 | `04d87ea4ba7f7a822c505c4ef58065b3d7fa57b0780860b6fee84be36679b495` |
| `outputs/encoder_incidence_mlp_edge_tsp21_s0_c5cddca1.pt` | 204413 | `d95a5c40da7c0bba4d16cc0b4df50ba94138218a6f9cf5837986086ce42e7e22` |
| `outputs/encoder_incidence_mlp_edge_tsp21_s1_c5cddca1.pt` | 204413 | `57063cf40bf3e83216a6ae4915ee79d13e3421eea57769f0a77280100d90b095` |
| `outputs/encoder_incidence_mlp_edge_tsp21_s2_c5cddca1.pt` | 204413 | `a0d573dac510aec3b75c58807430b8a6a29766241f924977c896ba2e08e7151e` |
| `outputs/encoder_incidence_mlp_edge_tsp99_s0_c5cddca1.pt` | 204413 | `9f2016d9ad005b9be1f6b8c40f4fe259c7db1a9fd399e9d8266eb183c307c355` |
| `outputs/encoder_incidence_mlp_elite_synthetic_tsp21_s0_c5cddca1.pt` | 204631 | `2cdc41ffa2b16af494f214efba147902a45ea2eb3e02cd37b1fcad2e906059b1` |
| `outputs/encoder_incidence_mlp_population_edge+local1-df1dfd0d_tsp21_s0_c5cddca1.pt` | 205623 | `f4ed242199537e57c235bef001a36f5cffdb36b9dcee2eef4c7382f5ac007b04` |
| `outputs/encoder_incidence_mlp_population_edge+local1-df1dfd0d_tsp21_s1_c5cddca1.pt` | 205623 | `04aa5ca2b523dd045fc3e67d267fadadd0aadd333aefaab75f0ce82a36a700e0` |
| `outputs/encoder_incidence_mlp_population_edge+local1-df1dfd0d_tsp21_s2_c5cddca1.pt` | 205623 | `54c8f0f303a719ef5450c40e28a57a599792531f811958d2e739b192643488f0` |
| `outputs/encoder_incidence_mlp_population_edge_tsp101_s0_c5cddca1.pt` | 204709 | `2b160d9c884e6390ccbfba90b0104c9a93106725cbd58b63db34592704da5b63` |
| `outputs/encoder_incidence_mlp_population_edge_tsp102_s0_c5cddca1.pt` | 204709 | `02ffb9ae128e3a3c958e39761cac92edd779d7dcd204d1e3cd0e7b2c41cf547b` |
| `outputs/encoder_incidence_mlp_population_edge_tsp103_s0_c5cddca1.pt` | 204709 | `f371daede82d492d6d721b77107bb74e47b7c32f64ed96b802af8946206957ea` |
| `outputs/encoder_incidence_mlp_population_edge_tsp104_s0_c5cddca1.pt` | 204709 | `5e4247fd167fa96e35640247f1bce487afbb3fff3c17170c500cc03b1f6f109e` |
| `outputs/encoder_incidence_mlp_population_edge_tsp105_s0_c5cddca1.pt` | 204709 | `3172cc5e06122b8bb4acbfab847f6a81cd4cf73d0817bac66926ef14cfb784e6` |
| `outputs/encoder_incidence_mlp_population_edge_tsp21_s0_c5cddca1.pt` | 204631 | `2dfe84bf353f7fbf0a4f82e8bbd688a98abcaa0c26eb8c2bfb18ae2a44009a15` |
| `outputs/encoder_incidence_mlp_population_edge_tsp21_s1_c5cddca1.pt` | 204631 | `03cb3135e2a9b0902c857f7e504a9c66c5eb67e66386fcdcdae81b82d63f40c2` |
| `outputs/encoder_incidence_mlp_population_edge_tsp21_s2_c5cddca1.pt` | 204631 | `1af2b4513b41997079b531d580950b66cbb589cc87bffe6d2380fb11175244f9` |
| `outputs/encoder_lstm_edge_tsp21_s0_c5cddca1.pt` | 437321 | `6b6b0480c82586d5818073f3d9d68b1057627fc955239de941ee6753602def49` |
| `outputs/encoder_lstm_edge_tsp21_s1_c5cddca1.pt` | 437321 | `f7407303c2ddf39f3c4b0e47b30610527e31551577810cb006db7e6b540090cc` |
| `outputs/encoder_lstm_edge_tsp21_s2_c5cddca1.pt` | 437321 | `8eafdc02af9ede766787acd9bafb07a01ba1b865a056d279920734e9637b25b1` |

### Training logs (per-epoch history next to each checkpoint)

| file | bytes | sha256 |
|---|---:|---|
| `outputs/encoder_edge_deepsets_edge_tsp21_s0_c5cddca1.json` | 20396 | `ae7dc4665c50396fb4f8140d6518290fd8921f54c30eabfe3b7b49fe20f1e666` |
| `outputs/encoder_edge_deepsets_edge_tsp21_s1_c5cddca1.json` | 20381 | `fecfc1b51ed297e48c9b9722c934d8500298e4f80e2cadeb9c0917627725880d` |
| `outputs/encoder_edge_deepsets_edge_tsp21_s2_c5cddca1.json` | 20397 | `0597325982d23a2ed939322a3304dd6101710319ab48c669faafccb81e186be9` |
| `outputs/encoder_edge_deepsets_edge_tsp99_s0_c5cddca1.json` | 20396 | `28d43092a890deb3c579a53b218d5d95d4756acbe6a9df755685e36a4ea50a3a` |
| `outputs/encoder_edge_deepsets_elite_synthetic_tsp21_s0_c5cddca1.json` | 20413 | `c30a1ea0b44a674befcd4aa41cce9706efcd584ed1dc096b04fed8a314952177` |
| `outputs/encoder_edge_deepsets_population_edge_tsp21_s0_c5cddca1.json` | 20373 | `0a6186fdfe73b755a4d62a020b268908cff35ea1ffef021bbec4335f8a32f600` |
| `outputs/encoder_incidence_mlp_edge+local0.25-df1dfd0d_tsp21_s0_c5cddca1.json` | 33215 | `4c101d9680a3af2997943b2843fb1f5842b24e25960b2a7b12756cba5a4b5afa` |
| `outputs/encoder_incidence_mlp_edge+local0.5-df1dfd0d_tsp21_s0_c5cddca1.json` | 33242 | `178bb1c29629f81fb2cf94a78b860e85b035c7e7fa8e222343fea0ed78e7dabe` |
| `outputs/encoder_incidence_mlp_edge+local1-df1dfd0d_tsp21_s0_c5cddca1.json` | 33237 | `7c4a71635bd6d9adac878148e15bf5b483127d617fc64069184aca65d1d87aed` |
| `outputs/encoder_incidence_mlp_edge+local1-df1dfd0d_tsp21_s1_c5cddca1.json` | 33256 | `b470ac0ccf74f2638b6ae76b31c79380d405b1bd5335557b80836224ae9a7689` |
| `outputs/encoder_incidence_mlp_edge+local1-df1dfd0d_tsp21_s2_c5cddca1.json` | 33259 | `2edf8cd18941ecd1410ca96f75a10ba9ff9bf9169b78b7ab90150f13127e3ab1` |
| `outputs/encoder_incidence_mlp_edge_tsp101_s0_c5cddca1.json` | 20410 | `b32efcb873bcdcea011967c395b1f5f395c825b9f03a85c77d1ed892ddb19b28` |
| `outputs/encoder_incidence_mlp_edge_tsp102_s0_c5cddca1.json` | 20408 | `7d60e53fa502c8fc3ad1685d2b2511104851449e5f00558574134b2e495ffcee` |
| `outputs/encoder_incidence_mlp_edge_tsp103_s0_c5cddca1.json` | 20405 | `84e820b0f42e546dbf0be99a14bd29a670675ccef118fa33cc3a44e63f9b6b25` |
| `outputs/encoder_incidence_mlp_edge_tsp104_s0_c5cddca1.json` | 20400 | `dc4505f938a5ab3a24ce81bcf4d02b124133604ddfe1e7b60c7fe98232e9bd7e` |
| `outputs/encoder_incidence_mlp_edge_tsp105_s0_c5cddca1.json` | 20395 | `d77eb93662538f282448edca957fdb6a73712711e4e741844070e603c6796148` |
| `outputs/encoder_incidence_mlp_edge_tsp21_s0_c5cddca1.json` | 20395 | `583e390b1d1ff5d3a15344b92f0849554f1fc3e7627b11f59ef610e208f91d5e` |
| `outputs/encoder_incidence_mlp_edge_tsp21_s1_c5cddca1.json` | 20423 | `1d846164cebdda050eb5e081644b2240a0dd5c283b00a056a92f3819d80e4512` |
| `outputs/encoder_incidence_mlp_edge_tsp21_s2_c5cddca1.json` | 20418 | `2bd562b24d00294039e67ca880321a85b377fd1e5bc6703ffb65c63a6e02eee7` |
| `outputs/encoder_incidence_mlp_edge_tsp99_s0_c5cddca1.json` | 20390 | `50f2830a4e6ebcf5cbde9d426b6f1f6f952c0e04a949d165c1763f2c4784a00a` |
| `outputs/encoder_incidence_mlp_elite_synthetic_tsp21_s0_c5cddca1.json` | 20482 | `631758a5f03f92bce2dec33bf20c34a9cf1257c5c3cfff62aa2a8dbaeaa17030` |
| `outputs/encoder_incidence_mlp_population_edge+local1-df1dfd0d_tsp21_s0_c5cddca1.json` | 33264 | `e6e4b995fb6bbef2373ed51f5adf0657ab481384f6602a070d31b455705eb62a` |
| `outputs/encoder_incidence_mlp_population_edge+local1-df1dfd0d_tsp21_s1_c5cddca1.json` | 33261 | `44641ae3453dd13fadd633212d0a37c2b8d234e324fef70db09b4779c5f5e0f0` |
| `outputs/encoder_incidence_mlp_population_edge+local1-df1dfd0d_tsp21_s2_c5cddca1.json` | 33268 | `1734fc724f9e49b10262fba882275169feeb68ae399a79bded7232f3c7a861fd` |
| `outputs/encoder_incidence_mlp_population_edge_tsp101_s0_c5cddca1.json` | 20396 | `063c5556981f510b7f8fd715319344b43ad40577735a8e8f0ca0ea4f1bccb7ad` |
| `outputs/encoder_incidence_mlp_population_edge_tsp102_s0_c5cddca1.json` | 20349 | `1d197b5ddf99ff41af3ec690fa3bab222a5b0261df14962b510beea5f3b5ebdb` |
| `outputs/encoder_incidence_mlp_population_edge_tsp103_s0_c5cddca1.json` | 20374 | `d4ccb814b57155cbb85d5881825315ee679eb52eb3bc50666fc69b273367b548` |
| `outputs/encoder_incidence_mlp_population_edge_tsp104_s0_c5cddca1.json` | 20375 | `834efd8adea4379056920110be127508e85211b64a822d7f367b8aa6b0ff48c4` |
| `outputs/encoder_incidence_mlp_population_edge_tsp105_s0_c5cddca1.json` | 20391 | `74bb39077c6c974d91da34873cbb08e0273211b40ac429e871bcae67f2b94af6` |
| `outputs/encoder_incidence_mlp_population_edge_tsp21_s0_c5cddca1.json` | 20408 | `b1b9867747ecb34095411b727ff991958ffcf02c23b99a0955049c56bd4cf84b` |
| `outputs/encoder_incidence_mlp_population_edge_tsp21_s1_c5cddca1.json` | 20391 | `bfb40a4eabb9953f0dd13630819f504d6d2dcfc8cb09d8cde46ff9964a61eb8a` |
| `outputs/encoder_incidence_mlp_population_edge_tsp21_s2_c5cddca1.json` | 20397 | `0f670ff0f9c654470a99ac9ccabf9b2fd7589fd03bfeb715b59b9cbd6ef1fa63` |
| `outputs/encoder_lstm_edge_tsp21_s0_c5cddca1.json` | 20402 | `0f80e63cdac55bb920644c94108dc74215c509f6ba47add339af8d073212c6c7` |
| `outputs/encoder_lstm_edge_tsp21_s1_c5cddca1.json` | 20398 | `d879b16f772ee9bea2db064b65458d113e9a7159f9312e9e61d30d0e71320c41` |
| `outputs/encoder_lstm_edge_tsp21_s2_c5cddca1.json` | 20410 | `32a7591ba144215929dcba33f4a77baedd7d90288e1c2b878443ef35bf935704` |

### v1 history (copies of the outputs of tag `v1-wyniki`, used only for the v1 row)

| file | bytes | sha256 |
|---|---:|---|
| `outputs/v1/trajectories.npz` | 1761004 | `167b0942018da044b9e2069c7f620636d8d83af93fa0d8e798f8b49c07896ca2` |
| `outputs/v1/encoder_variant_trajectory.pt` | 437177 | `504cc44c24305ae1168abc7d8c56e9fd075fada03e38a96eacc8cc4c5265b463` |
| `outputs/v1/encoder_variant_edge_canonical_100.pt` | 437273 | `82cf8197fad499fb1d422fddacc7cd2f1216cddafa28029bb92530865810b110` |

### Unchanged control

| file | bytes | sha256 |
|---|---:|---|
| `outputs/stn_pca.png` | 183281 | `0ef89ece117d3ee0247e4447372813cf9cef384e1080c99359e006d3e0d87a2b` |
