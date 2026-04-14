# Dependency and environment pins (reproduction)

Record **exact** versions used for the **published numbers** where they differ from broad manuscript wording (e.g. Data Availability Statement “ANDES 1.9.x”).

## Paper repository slug (public GitHub name)

**`PAPER_TITLE_SLUG`:** `unified-physics-informed-neural-surrogate-smib-tsa-cct-benchmark`

Derived from the IEEE Access manuscript title (filesystem-safe, GitHub-safe length). If the name is taken on GitHub, append a short suffix (e.g. `-ieee-access-2026`) and update this file and the Data Availability Statement.

## Python

- **Recommended for paper-aligned ANDES 1.9.x:** Python **3.8–3.11** (match ANDES 1.9.x docs).
- **If using ANDES 1.10.x:** Python **≥ 3.10** (see `docs/guides/ANDES_VERSION_INFO.md` in the development monorepo).

## PyTorch (CPU-first)

- Install the **CPU** wheel from [pytorch.org](https://pytorch.org/get-started/locally/) for your Python version.
- Pin in prose what you used for the paper, e.g. `torch==2.x.x+cpu` (fill from `pip freeze` on the authoring machine).
- **Training device:** set `training.device: cpu` in YAML for reproduction on CPU-only machines.

## ANDES (patch level for reviewers)

- **Manuscript (broad):** ANDES **1.9.x** is acceptable in the Data Availability Statement.
- **Reproduction docs (specific):** pin the **patch** used when generating paper trajectories, e.g. **`andes==1.9.3`** (update if your authoring run used another 1.9.x patch).
- **Note:** the development machine may show a newer ANDES (e.g. 1.10.0); Path B **does not** require ANDES if you only consume Zenodo CSVs. Path A and sim-backed CCT scripts **do** require ANDES; use the **same patch** as documented here when claiming parity with sim-backed numbers.

## Random seeds and threads

- YAML `reproducibility.random_seed` (e.g. **42**) in `smib_pinn_ml_matched_pe_direct_7_parity_dropout_wd_pinn_no_residual.yaml`.
- Optional: set `OMP_NUM_THREADS` / `MKL_NUM_THREADS` to a fixed value for more stable CPU linear algebra across hosts.

## Numerical tolerance vs bit-identical

Retraining with Tier 1 CSVs can yield **last-digit differences** in metrics vs the PDF due to BLAS/thread order and PyTorch. **Bit-identical** reproduction is **not** claimed for full retrains. Use **Tier 2** (checkpoints) + **Tier 3** (frozen JSON) to audit headline numbers; interpret agreement within **reasonable float tolerance** unless checksum-matched JSON is supplied.

## Zenodo record

- **Tier 1:** `data/processed/exp_20260211_190612/` (~**237 MB** in the authoring snapshot — host on Zenodo, not in public Git). **Minimal footprint:** at minimum the paired **train / val / test** CSVs for this experiment id; strongly recommended in the same folder: `preprocessing_provenance.json`, `all_splits_*.csv` if they exist. Do **not** place unrelated monorepo data in this record.
- **Tier 2:** validation-selected PINN* / Std NN* checkpoints and selection metadata (optional; for headline metric audit without re-running selection).
- **Tier 3:** `comparison_results.json`, `pe_input_cct_test_results.json`, etc., cited in `paper_outputs_index.md` (optional).

**Path A then Path B:** full ANDES regeneration (**Path A**) does **not** require Zenodo Tier 1 beforehand; **Path B** then trains on the new `data/processed/...` folder. If you compare Path A outputs to Zenodo Tier 1, expect only **scientific** agreement, not guaranteed numeric identity (see `README.md` and `reproduction_steps.md`).

Placeholder DOI until published: `10.5281/zenodo.XXXXXXX` (replace in README and Data Availability Statement).

## Requirements files (public repo)

Ship copies of:

- `requirements.txt`
- `requirements_pinn.txt`
- `requirements-andes.txt` (optional pointer for ANDES install)

from the development monorepo root at public repo root.
