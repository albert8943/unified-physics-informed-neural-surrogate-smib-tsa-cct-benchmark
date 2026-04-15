# Export notes (maintainer)

This tree was produced by `public_reproduction/scripts/export_ieee_access_public_repo.ps1` from the private development monorepo.

- `scripts/` and `configs/` are subset exports: paths from `paper_outputs_index.md` plus transitive `from scripts.?? imports (`resolve_paper_export_paths.py`). Add backticks in that index when new drivers should ship publicly.
- Do **not** commit `data/` or large `outputs/`; Tier 1 splits live on **Zenodo** (see root `README.md` / `dependency_versions.md`), e.g. `https://doi.org/10.5281/zenodo.19562416`.
- To publish: create an **empty** GitHub repo, then from this folder:
  `git remote add origin https://github.com/<USER>/unified-physics-informed-neural-surrogate-smib-tsa-cct-benchmark.git`
  `git push -u origin main`
