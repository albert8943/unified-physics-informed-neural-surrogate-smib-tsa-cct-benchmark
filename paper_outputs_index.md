# Paper outputs index (scripts, configs, data)

Maps **manuscript artifacts** (tables, figures, analyses) to **driver scripts**, **primary YAML**, **archived outputs** (paths as in the development repo / Zenodo tiers), and whether **ANDES** is required. LaTeX labels refer to `text/5_Results_and_Discussion.tex` in the IEEE Access template.

**Primary config (PINN* parity, plain MLP):** `configs/experiments/smib/smib_pinn_ml_matched_pe_direct_7_parity_dropout_wd_pinn_no_residual.yaml`  
**Processed split (Tier 1 Zenodo):** `data/processed/exp_20260211_190612/` (train/val/test CSVs)

| Manuscript focus | Script(s) | Config / inputs | Primary output(s) (examples) | Simulator? | Zenodo tier |
|------------------|-----------|-----------------|------------------------------|--------------|-------------|
| Path B retrain / full experiment (Track 1 in `reproduction_steps.md`) | `scripts/run_complete_experiment.py` | Primary YAML + `--data-dir` Tier 1 or Path A splits | under `outputs/` | PyTorch; ANDES only if data generation is not skipped | 1 + **2** |
| Fair comparison test table, trajectory RMSE, paired \(p\) | `scripts/compare_models.py` | Headline PINN* / Std NN* checkpoints; test split CSVs | `outputs/expt_residual_backbone_retrain_20260407/compare_test/pinn_nores_vs_mlstar/comparison_results.json`; also `outputs/campaign_indep_final_test_20260406/comparison_results.json` | PyTorch if eval only | 1 + **2** |
| Trajectory comparison PDFs | `scripts/compare_models.py` (regenerate) | same | `figures/trajectory_comparison/model_comparison_delta_{stable,unstable}.pdf` (manuscript paths) | PyTorch if from checkpoints | 2 |
| SMIB stats table (per-scenario RMSE) | `scripts/compare_models.py` → JSON stats | same comparison JSON | `comparison_results.json` | PyTorch | 2 / 3 |
| Pe-input CCT test tables / headline CCT | `scripts/run_pe_input_cct_test.py` | PINN* (and baselines) paths per campaign | e.g. `outputs/paper_pe_input_cct_sensitivity_pw/pw_0p05/pe_input_cct_test_results.json`; `outputs/paper_pe_input_cct_rerun_20260411/pe_input_cct_test_results.json` | **ANDES** if script re-sims | 2 / 3 |
| CCT LaTeX only from frozen JSON | `scripts/run_pe_input_cct_test.py --from-json … --print-latex-tables` | JSON path | stdout / captured `.tex` | No (JSON-only) | **3** |
| Persistence window sweep (CCT vs \(\Delta t_w\)) | `scripts/run_pe_input_cct_persistence_window_sensitivity.py` | configs + model paths | `outputs/paper_pe_input_cct_sensitivity_pw/` | **ANDES** if re-query sim | 2 |
| Validation \(\Delta t_w\) sweep + selection | `scripts/select_delta_tw_from_val_cct_sweep.py` | val sweep outputs under `outputs/pe_input_cct_val_pw_sweep/` | selected window for headline | **ANDES** if sweep uses sim | 2 |
| Pe-input CCT figures | `scripts/plot_pe_input_cct_paper_figures.py` | JSON + style paths | `paper_writing/IEEE Access Template/figures/pe_input_cct/*.png` | No if plotting from files | 3 |
| Benchmark ensemble figures | `scripts/plot_benchmark_ensemble_paper_figures.py` | `--data` all splits or trajectory CSV | `figures/benchmark_ensemble/*.pdf` | No if from CSV | 1 |
| Multiseed \(\delta\) campaign | `scripts/run_multiseed_delta_campaign.py` | campaign configs | `docs/publication/pre_specs/multiseed_delta_rmse_results.json` | **ANDES** if regenerating | 1–2 |
| Validation sweeps (PINN \(\lambda\), ML hyperparams) | internal sweep drivers / docs | `docs/publication/pre_specs/phase2_*.json`, `phase3_*.json` | `phase4_test_comparison_results_20260407.json` | Mixed | 2 |
| Swing residual diagnostics | `scripts/compute_swing_residual_diagnostics.py` | headline PINN* / ML* paths | `docs/publication/pre_specs/swing_residual_diag_valgate_test.json`; ANDES truth: `swing_residual_diag_andes_truth_only_20260407.json` | **ANDES** for truth path | 2 / 3 |

**Legend:** **Tier 1** = processed CSVs; **Tier 2** = checkpoints + selection artifacts; **Tier 3** = frozen JSON (and optional pre-rendered assets). See `README.md` for **Path A** (upstream, ANDES) then **Path B** (downstream, splits from Zenodo or from Path A) and how they use these tiers.
