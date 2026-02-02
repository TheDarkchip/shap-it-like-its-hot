# Project Roadmap: Big SHAP Energy

This repository implements a compact, reproducible case study on the Statlog German Credit dataset to understand when global SHAP feature importances are stable and how they compare to permutation feature importance (PFI), especially under class imbalance and (optionally) feature correlation.

Last updated: 2026-02-01

## At a glance

- Objective: stability of global explanations under controlled training imbalance, plus SHAP vs PFI agreement.
- Dataset: Statlog German Credit (single dataset).
- Model: XGBoost trees (TreeSHAP).
- Importance methods: mean(|SHAP|) and PFI.
- Evaluation: repeated nested cross-validation with inner HPO and untouched outer test folds.
- Outputs: fold-level importance vectors, stability and agreement metrics, and uncertainty summaries.
- Selected bounded novelty stretches (post-MVS): PFI metric sensitivity + signed SHAP directionality for top features.

## Table of contents

1. Scope and boundaries
2. Research questions
3. Prior-work check (deliverable)
4. Evaluation protocol
5. Metrics
6. Experiment plan
7. Feedback checkpoints
8. Work packages and acceptance criteria
9. Proposed repository layout
10. Risks and mitigations

## 1) Scope and boundaries

Primary objective: quantify (a) stability of global importances within a method and (b) agreement between SHAP and PFI, under controlled changes in training imbalance on a small credit dataset.

Scope is now locked for the minimum viable study (MVS). Any expansions (extra datasets, many more explainers, heavy ablations) are explicitly treated as optional stretch work.

Default plan (minimum viable study, MVS):
- Dataset: Statlog German Credit (fixed, single dataset).
- Primary model: XGBoost tree model + TreeSHAP.
- Imbalance settings: 10%, 30%, 50% positives in the training fold (test fold untouched).
- Feature importance methods:
  - global SHAP via mean absolute SHAP
  - PFI via permutation importance (multiple repeats)
- Evaluation: nested CV with inner HPO and repeated outer splits for uncertainty.

Non-goals (out of scope unless explicitly requested):
- additional datasets
- many explainability methods beyond SHAP and PFI
- exhaustive HPO or large neural models
- causal claims about true feature importance

## 2) Research questions (RQ)

RQ1 - Stability under class imbalance
How stable are global SHAP importances (and PFI importances) when the effective training class ratio is varied, while evaluation is on untouched held-out folds?

RQ2 - SHAP vs PFI agreement under imbalance (rank + magnitude)
Across the same imbalance settings, when do SHAP and PFI agree or disagree on important features, and is disagreement primarily about ranking or magnitude (especially in near-flat importance regimes)?

RQ3 - Correlated features (optional sanity check)
What happens to SHAP and PFI global importances when a few features are duplicated into near-copies (high correlation)? Do methods split importance, concentrate it, or change combined importance? Does grouping correlated pairs improve agreement?

## 3) Prior-work check (explicit deliverable)

Before implementing the full experiment grid, the project verifies whether the main questions and experimental setup are already directly answered in closely related papers (especially imbalance vs explanation stability and SHAP/PFI disagreement on German Credit).

Deliverables:
- `docs/prior_work_check.md`: short synthesis, overlap/gap table, and final scope decisions.
- `docs/prior_work_summaries.md`: per-source summaries plus an explicit mapping to RQ1–RQ3.

Acceptance criterion: after this check, the experiment design is justified as a replication/extension with clear added value, and 1–2 bounded novelty stretches are selected (or explicitly deferred).

## 4) Evaluation protocol (nested CV with repeated outer splits)

Terminology:
- Nested CV: hyperparameters are chosen only using training data of each outer split via an inner CV loop; evaluation happens on the outer test fold.
- Repeated outer CV: multiple outer split realizations to estimate uncertainty.

Concrete protocol for one setting (one model + one class ratio):

Outer loop (repeated stratified K-fold):
1) Split data into (train_fold, test_fold) with stratification.
2) Resample only train_fold to reach the target class ratio (keeping fold size fixed).
3) Inner loop: HPO on the resampled train_fold using inner stratified CV.
4) Retrain with selected hyperparameters on the full resampled train_fold.
5) Evaluate performance on the untouched test_fold.
6) Compute explanations on test_fold:
   - PFI: permutation importance with ROC AUC scoring and multiple repeats.
   - SHAP: per-instance SHAP on test_fold, summarized as mean absolute SHAP per feature.
7) Store metrics and metadata (fold id, repeat id, random seed, chosen hyperparameters, runtime).

After all outer folds:
- Aggregate mean and uncertainty (standard deviation and confidence intervals from outer repeats).
- Compute stability metrics within each method (RQ1).
- Compute agreement metrics between methods (RQ2).

Early sanity checks:
- leakage check: no resampled duplicate from train_fold appears in test_fold
- reproducibility: fixed random seeds and deterministic settings where practical
- flat-importance check: detect near-uniform importances and interpret rank metrics carefully

## 5) Metrics (rank + magnitude, with guardrails)

Inputs per outer fold:
- SHAP importance vector s (mean absolute SHAP per feature on test_fold)
- PFI importance vector p (mean performance drop after permutation on test_fold)

Normalization:
- s_norm = s / sum(s), p_norm = p / sum(p) (if sums are non-zero)

Stability within a method (RQ1):
- Rank stability: average pairwise Spearman (or Kendall tau) between fold rankings.
- Magnitude stability: dispersion of s_norm or p_norm across folds (per-feature variance and average variance; optional CoV per feature).

Agreement between SHAP and PFI (RQ2):
- Rank agreement: Spearman correlation of ranks; optional Kendall tau.
- Top-k overlap: overlap size (or Jaccard) for k in {3, 5, 10}.
- Magnitude agreement: cosine similarity between s_norm and p_norm, plus L1 distance (or Jensen-Shannon distance).
- Magnitude differences for important features: for the top-k union, report |s_norm_j - p_norm_j| distribution.

Interpretability guardrail:
- Report an importance dispersion summary (entropy or Gini of s_norm or p_norm). When dispersion is low (importances nearly flat), emphasize magnitude metrics and top-k overlap over rank correlation.

Optional (selected) reporting add-ons:
- Signed SHAP: mean signed SHAP for top-k features + sign agreement across folds/settings.
- PFI metric sensitivity: recompute PFI with alternate scoring metrics and compare stability/agreement.

## 6) Experiment plan (MVS first, then bounded stretches)

Minimum viable study (default deliverable; required):
- Model: XGBoost trees.
- Class ratios: 10%, 30%, 50% positives in resampled training folds.
- Evaluation: repeated nested CV with modest HPO budget.
- Outputs: performance metrics; SHAP and PFI importances per fold; stability and agreement analyses for RQ1–RQ2.

Selected bounded novelty stretches (do after MVS; commit):
- D) PFI metric sensitivity under imbalance:
  - Recompute PFI using 2–3 metrics (ROC AUC, PR AUC, log-loss) on the same models/splits.
  - Report how stability (RQ1) and agreement (RQ2) vary with metric choice.
- F) Directionality / sign agreement (lightweight):
  - For top-k features, report mean signed SHAP alongside mean(|SHAP|).
  - Track sign agreement and (optional) signed-rank agreement.

Optional stretches (only if time and the MVS results merit it):
- A) Stronger HPO budget (compute permitting): SMAC more trials or Sobol sampling.
- B) Correlated-feature duplication ablation (RQ3 sanity check): duplicate 3–5 features as near-copies and see whether methods split vs concentrate importance; optionally test grouped importances.
- C) Linear baseline (robustness): add logistic regression or gblinear on the same MVS grid.

Deferred for now (bounded but not committed):
- E) Within-fold bootstrap CIs for global importance vectors (extra compute; only if needed after initial results).

## 7) Feedback-session checkpoints

Checkpoint 1 (early, before long runs): scope + protocol sign-off
- Confirm MVS scope is acceptable.
- Walk through nested CV pseudocode and confirm it matches expectations.
- Confirm uncertainty summaries (outer repeats vs single outer split).
- Confirm HPO budget and metric.

Checkpoint 2 (after first results): results sanity check
- Validate trends and confidence intervals.
- Confirm agreement metrics are interpreted correctly (especially when importances are flat).
- Decide whether optional stretches are worthwhile.

## 8) Work packages and acceptance criteria

WP0 - Prior work verification
- Outputs: `docs/prior_work_check.md`, `docs/prior_work_summaries.md`.
- Done when: scope is locked, overlap/gap is documented, and novelty stretches D + F are confirmed as post-MVS add-ons (with others explicitly deferred/optional).

WP1 - Repository scaffolding and data pipeline
- Implement data loading, preprocessing, and one end-to-end pipeline.
- Done when: a single run finishes and writes a results artifact.

WP2 - Evaluation harness (nested CV)
- Implement repeated outer CV and inner HPO.
- Add protocol tests: leakage checks, deterministic seeds, metric sanity.
- Output: `docs/protocol.md` with pseudocode and design notes.
- Done when: a small smoke test matches expectations.

WP3 - MVS experiments
- Run the full MVS grid and generate plots/tables.
- Output: `results/` with fold-level outputs and summary artifacts.
- Done when: summary plots for RQ1–RQ2 are generated.

WP4 - Stretches (selected + optional)
- Selected: Novelty D and F.
- Optional: A/B/C.
- Done when: each executed stretch has a short summary, plots, and “what changed” notes.

WP5 - Reporting and packaging
- Write up findings and limitations; include reproducibility notes.
- Output: `docs/report.md` (or final report PDF elsewhere), plus reproduction instructions.
- Done when: a fresh clone can reproduce core tables/figures from a single command.

## 9) Proposed repository layout

- `src/`
  - `data/` (loading, resampling utilities)
  - `models/` (train wrappers, parameter spaces)
  - `eval/` (nested CV harness, metrics)
  - `explain/` (SHAP and PFI computation)
  - `utils/` (seeding, logging, IO)
- `configs/` (YAML/JSON for experiment settings)
- `scripts/` (run_mvs.py, run_ablation.py, summarize.py)
- `results/` (ignored by git; fold-level outputs)
- `docs/` (protocol notes, prior work check, interpretation notes)

## 10) Risks and mitigations

Risk: Too many combinations; runs take longer than expected.
- Mitigation: lock MVS first; keep HPO budget small; run optional stretches only if MVS is complete.

Risk: Validation protocol is misunderstood or leaky.
- Mitigation: WP2 protocol doc + sanity tests; review in checkpoint 1.

Risk: Rank metrics are noisy when importances are flat.
- Mitigation: always report magnitude-aware metrics and dispersion.

Risk: Permutation importance variance is high.
- Mitigation: multiple permutation repeats; report uncertainty across folds; optional within-fold bootstrap only if needed.

Risk: Stretch goals expand scope too much.
- Mitigation: keep selected stretches bounded (reuse MVS runs), and keep everything else explicitly optional/deferred.
