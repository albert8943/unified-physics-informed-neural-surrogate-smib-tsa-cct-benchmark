# Step-by-step reproduction (IEEE Access SMIB)

All commands assume the **repository root** of the **public code repository** (the folder created by `git clone …/unified-physics-informed-neural-surrogate-smib-tsa-cct-benchmark.git`). On Windows PowerShell, use the same commands unless noted.

## Overview: Path A and Path B

| | **Path A** | **Path B** |
|---|------------|------------|
| **Goal** | **Regenerate** trajectories and processed splits with **ANDES** from the published YAML recipe (full upstream pipeline). | Train/evaluate on **fixed** train/validation/test CSVs—either from **Zenodo Tier 1** or from **outputs of Path A** if you regenerated locally. |
| **ANDES** | **Required** for data generation. | Not required to *train* on Tier 1 CSVs alone; still required for some downstream scripts (see dependency table). |
| **Zenodo** | Optional for comparison; regenerated CSVs may **differ numerically** from Tier 1. | **Tier 1** recommended for the exact paper splits (~237 MB); **Tier 2/3** for headline metrics / audits. |
| **When to use** | Audit or extend the **simulation side**; build splits without downloading Zenodo. | Match paper workflow on **released** supervised data; fastest for most reviewers. |

**Documentation order:** **Path A** (simulation → data) is described **first**, then **Path B** (supervised learning on splits). Path B is independent of Path A if you use Zenodo Tier 1; if you run Path A first, point training at your generated `data/processed/...` folder instead of the Zenodo folder name when paths differ.

---

## 1. Clone the public repository (Path A and Path B)

```bash
git clone https://github.com/<your_github_user>/unified-physics-informed-neural-surrogate-smib-tsa-cct-benchmark.git
cd unified-physics-informed-neural-surrogate-smib-tsa-cct-benchmark
```

---

## 2. Python environment (CPU-first) (Path A and Path B)

- **Python:** 3.8+ (3.10+ if you align with ANDES 1.10.x; paper DAS references ANDES 1.9.x — see `dependency_versions.md` for the **exact ANDES patch** used for sim-backed steps).
- **PyTorch (CPU):** install the **CPU** build from the [official PyTorch install page](https://pytorch.org/get-started/locally/) (choose CPU, your OS, pip/conda). Do **not** assume CUDA.
- **Dependencies:** from repo root:

  ```bash
  pip install -r requirements.txt
  pip install -r requirements_pinn.txt
  ```

- **ANDES:** install separately for **Path A** or any step marked **ANDES** in the dependency table (e.g. `pip install andes==<patch>` — pin matches `dependency_versions.md`).

**Force CPU for training (Path B):** in the primary YAML, set `training.device: cpu` (not `auto`, which may select GPU if installed). Alternatively document `CUDA_VISIBLE_DEVICES=` empty only for users who have a GPU build installed but want CPU.

**Threads (optional):** if you need reproducible CPU numerics across machines, document fixed `OMP_NUM_THREADS` / `MKL_NUM_THREADS` (see `dependency_versions.md`).

---

## Path A — Regenerate data with ANDES (optional; upstream of Path B)

Path A is a **full pipeline replay** with ANDES. It is **not** claimed to recover **byte-identical** trajectories to Zenodo Tier 1 CSVs unless ANDES patch, solver settings, time step, sampling chain, and platform are fully pinned. For the **same supervised splits as the paper** without regenerating, use **Path B** with Zenodo Tier 1.

### A.1. Sanity check (before long runs)

```bash
python -c "import andes; print(andes.get_case('smib/SMIB.json'))"
```

If this fails or prints an unexpected path, fix the ANDES install or bundled examples before continuing.

### A.2. Regenerate and preprocess

Follow internal publication command references in the development repo: `docs/publication/QUICK_REFERENCE_COMMANDS.md` (not shipped in minimal public export unless you copy it). Use the same primary YAML with **data generation enabled** and ANDES **patch-pinned** per `dependency_versions.md`. After preprocessing, note the `data/processed/exp_*` directory your run creates; you will pass it as `--data-dir` in Path B if you skip Zenodo Tier 1.

---

## Path B — Supervised learning on fixed splits (downstream of Path A or Zenodo)

Path B reproduces **supervised learning and evaluation** on fixed train/validation/test CSVs. Either use **Zenodo Tier 1** (same `exp_20260211_190612` layout as the paper) **or** CSVs produced by **Path A** above.

Path B alone does **not** re-run ANDES to rebuild Zenodo’s Tier 1 trajectories; it consumes CSVs that already exist (released or locally generated).

### B.1. Obtain splits

**Option 1 — Zenodo (recommended for paper-aligned splits):**

1. Open the **Zenodo DOI** from the paper’s Data Availability Statement (placeholder: `https://doi.org/10.5281/zenodo.XXXXXXX`).
2. Download the `.zip` archive.
3. Extract so that this path exists **relative to the repo root**:

   `data/processed/exp_20260211_190612/`

   (The Zenodo bundle should use top-level folder `unified-physics-informed-neural-surrogate-smib-tsa-cct-benchmark/` per publication packaging; place `data/` beside `scripts/`, `pinn/`, etc., or merge into the clone so the relative path above resolves.)

**Option 2 — After Path A:** use the `data/processed/<your_exp_id>/` path written by your regeneration run as `--data-dir` below.

### B.2. Two tracks after splits exist

**Track 1 — Retrain and re-select**

```bash
python scripts/run_complete_experiment.py ^
  --config configs/experiments/smib/smib_pinn_ml_matched_pe_direct_7_parity_dropout_wd_pinn_no_residual.yaml ^
  --skip-data-generation ^
  --data-dir data/processed/exp_20260211_190612
```

(Use `\` continuation on Unix.) Replace `--data-dir` with your Path A output folder if applicable. Add any flags your parity recipe requires (e.g. `--lambda-physics`, skip flags, partner checkpoint paths) exactly as in the **comment block** at the top of that YAML in the **development** monorepo.

**Warning:** test metrics may **differ** from the manuscript unless validation selection is replayed exactly. Treat agreement as **within floating-point tolerance**, not bit-identical, unless you ship frozen JSON (Tier 3).

**Track 2 — Paper-aligned headline test metrics (Tier 2 + optional Tier 3)**

Use validation-selected **checkpoint paths** and **evaluation/compare-only** commands documented in the YAML comments and `paper_outputs_index.md`. Tier 2 is the **default expectation** for matching reported **test** performance without rerunning the full validation campaign.

---

## Step × dependency (PyTorch vs ANDES)

| Step / script (representative) | PyTorch only | ANDES required |
|--------------------------------|--------------|----------------|
| Training / eval on CSVs (`run_complete_experiment.py`, `train_model.py`) | Yes | No |
| `scripts/compare_models.py` (on saved predictions/checkpoints) | Yes | No |
| `scripts/run_pe_input_cct_test.py` (if re-simulates reference CCT) | Partial | **Yes** (reference trajectories / clearing) |
| `scripts/run_pe_input_cct_persistence_window_sensitivity.py` | Partial | **Yes** if ANDES back-end used |
| `scripts/select_delta_tw_from_val_cct_sweep.py` | Partial | **Yes** if sweep uses sim |
| `scripts/compute_swing_residual_diagnostics.py` (truth from ANDES) | Partial | **Yes** when comparing to sim truth |
| `scripts/plot_pe_input_cct_paper_figures.py` / `plot_benchmark_ensemble_paper_figures.py` | Yes (given inputs) | No if inputs are files only |
| `scripts/run_multiseed_delta_campaign.py` | Depends | **Yes** if regenerating data |
| Path A data generation | No | **Yes** |

Case file in YAML: `smib/SMIB.json` → resolved via **`andes.get_case`** in the codebase.

---

## Verify-only path (Tier 3 JSON)

If Zenodo includes frozen `pe_input_cct_test_results.json` (or similar), regenerate LaTeX fragments without training, e.g.:

```bash
python scripts/run_pe_input_cct_test.py --from-json <path_to_json> --print-latex-tables
```

(No ANDES where the script operates JSON-only.)

---

## Numerical identity

Published metrics should be read as consistent within **typical floating-point tolerance** for retrains; **bit-identical** reproduction is **not** claimed unless you verify against frozen Tier 3 JSON with checksums. Pin **Python + PyTorch (CPU)** as in `dependency_versions.md`.
