# IEEE Access SMIB — public reproduction materials



**Paper title:** Unified Physics-Informed Neural Surrogate Model for Transient Stability Assessment and Trajectory-Based Critical Clearing Time Search on a Single-Machine Infinite-Bus Benchmark



This directory is **staging documentation** inside the development repository. The same files are intended to be copied to the root of the **separate public GitHub repository** named `unified-physics-informed-neural-surrogate-smib-tsa-cct-benchmark` (see `dependency_versions.md` for the slug). **Large frozen datasets are not stored in Git**; they are published on **Zenodo** (see below).



## Scope boundary (Paper A vs private monorepo)

- **Public GitHub:** minimal runnable **Paper A** tree only (fresh history; no monorepo `.git`). See [`../MAINTAINERS_export_to_public_repo.md`](../MAINTAINERS_export_to_public_repo.md) for the copy list and pre-push checks.

- **Public Zenodo:** **Tier 1** processed splits required for this paper’s Path B (see `dependency_versions.md`). Tier 2/3 are optional for auditing headline metrics.

- **Private:** the full **development monorepo** (all other code, data, `paper_writing/`, future experiments) is **not** pushed to the public GitHub remote; future papers use their own curated exports.



## Two reproduction paths (Path A and Path B)



Reproduction follows the **same causal order as the study**: **Path A** (ANDES → trajectories → processed splits), then **Path B** (supervised learning and evaluation on those splits). In practice many readers **only** run **Path B** using the **released** Zenodo splits, skipping Path A.



- **Path A (optional; upstream):** reproduces the **end-to-end workflow** including **ANDES** trajectory generation and preprocessing from the published YAML settings. It is for a **scientifically equivalent** re-run of the data pipeline; it is **not** claimed to yield **byte-identical** CSVs to Zenodo Tier 1 unless every simulator and sampling detail matches the original run. See `reproduction_steps.md` for the `andes.get_case` sanity check and ANDES patch pins.



- **Path B (downstream):** reproduces **supervised learning and evaluation on fixed train/validation/test CSVs**—either the **Zenodo Tier 1** bundle matching the paper or **new CSVs from Path A**. Path B does **not** re-run ANDES to recreate Zenodo’s Tier 1 trajectories; it trains on CSVs that already exist. **Tier 2/3** on Zenodo support headline metrics and audits as described below.



## Where to get code vs data



| What | Where | Action |

|------|--------|--------|

| **Code** (scripts, configs, packages) | **Public GitHub repository** (same slug as above; fresh history — see maintainer export guide) | **Fork or clone** for software |

| **Data** (~237 MB Tier 1 processed splits, optional Tier 2 checkpoints, Tier 3 JSON) | **Zenodo** | **Download** when using **Path B** with paper-frozen splits (DOI in the Data Availability Statement; placeholder: `https://doi.org/10.5281/zenodo.XXXXXXX` — replace with your record) |



**Workflow (Path A then Path B, full stack):** clone the public GitHub repo → install ANDES at the pinned patch → run **Path A** in `reproduction_steps.md` to generate `data/processed/...` → run **Path B** pointing `--data-dir` at that folder (or switch to Zenodo Tier 1 for paper-identical splits).



**Workflow (Path B only, typical for reviewers):** clone → download Zenodo → extract `data/processed/exp_20260211_190612/` → run **Path B** training/evaluation in `reproduction_steps.md` (Path A skipped).



## Hardware (CPU-first)



All authoring for this paper used **CPU-only** training and evaluation (no GPU required for reproduction as documented). Optional GPU acceleration is not described as required.



## Headline test metrics (PINN* / Std NN*)



Applies to **Path B**. Reported **test** metrics use **validation-selected** checkpoints. To match headline tables **without** re-running the full selection campaign, use **Zenodo Tier 2** (checkpoints and/or selection logs). Tier 1 CSVs alone support retraining; numbers may differ from the PDF unless selection is replayed exactly (see `dependency_versions.md` for tolerance vs bit-identical).



## Guides in this folder



- **`reproduction_steps.md`** — **Path A** then **Path B** (clone, shared environment, ANDES regen, Zenodo or Path-A splits, two tracks, dependency table).

- **`paper_outputs_index.md`** — which tables/figures map to which scripts, configs, outputs, simulator need, Zenodo tier.

- **`dependency_versions.md`** — pinned Python, PyTorch (CPU), ANDES patch, seeds, threads, slug.



## Maintainer note



Export instructions for creating the **separate public Git repo** (without monorepo history) live one level up: [`../MAINTAINERS_export_to_public_repo.md`](../MAINTAINERS_export_to_public_repo.md).

