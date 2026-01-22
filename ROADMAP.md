# Project Roadmap: Big SHAP Energy

This repository implements a compact, reproducible case study on the Statlog German Credit dataset to understand when global SHAP feature importances are stable and how they compare to permutation feature importance (PFI), especially under class imbalance and (optionally) feature correlation.

Last updated: 2026-01-21

## At a glance

- Objective: stability of global explanations under controlled training imbalance, plus SHAP vs PFI agreement.
- Dataset: Statlog German Credit (single dataset).
- Model: XGBoost trees (TreeSHAP).
- Importance methods: mean(|SHAP|) and PFI.
- Evaluation: repeated nested cross-validation with inner HPO and untouched outer test folds.
- Outputs: fold-level importance vectors, stability and agreement metrics, and uncertainty summaries.

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

Default plan (minimum viable study, MVS):
- Dataset: Statlog German Credit (fixed, single dataset).
- Primary model: XGBoost tree model + TreeSHAP.
- Imbalance settings: 10%, 30%, 50% positives in the training fold (test fold untouched).
- Feature importance methods:
  - global SHAP via mean absolute SHAP
  - PFI via permutation importance
- Evaluation: nested CV with inner HPO and repeated outer splits for uncertainty.

Non-goals (out of scope unless explicitly requested):
- additional datasets
- many explainability methods beyond SHAP and PFI
- exhaustive HPO or large neural models
- causal claims about true feature importance

Stretch goals (optional; only after MVS is complete):
- slightly richer SMAC HPO budget
- small correlated-feature duplication stress test
- linear-model baseline (robustness)
- one bounded novelty stretch from Section 6

## 2) Research questions (RQ)

RQ1 - Stability under class imbalance
How stable are global SHAP importances (and PFI importances) when the effective training class ratio is varied, while evaluation is on untouched held-out folds?

RQ2 - SHAP vs PFI agreement under imbalance (rank + magnitude)
Across the same imbalance settings, when do SHAP and PFI agree or disagree on important features, and is disagreement primarily about ranking or magnitude (especially in near-flat importance regimes)?

RQ3 - Correlated features (optional sanity check)
What happens to SHAP and PFI global importances when a few features are duplicated into near-copies (high correlation)? Do methods split importance, concentrate it, or change combined importance? Does grouping correlated pairs improve agreement?

## 3) Prior-work check (explicit deliverable)

Before implementing the full experiment grid, the project verifies whether the main questions and experimental setup are already directly answered in closely related papers (especially imbalance vs explanation stability and SHAP/PFI disagreement on German Credit).

Deliverable: `docs/prior_work_check.md` containing:
- a 1-2 page summary of the most relevant 8-12 sources
- what each paper answers (and does not answer) relative to RQ1-RQ3
- final scope decisions (what we keep, drop, or adjust)
- 3-6 novelty stretch candidates that are not the central focus of the closest-overlap papers

Acceptance criterion: after this check, the experiment design is either (a) justified as a replication/extension with clear added value, or (b) narrowed further.

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
- Aggregate mean and uncertainty (standard deviation and bootstrap CIs).
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

Optional metrics (novelty stretches):
- Signed agreement: sign agreement and signed-rank agreement on top-k using mean signed SHAP.
- Distributional stability: compare per-feature SHAP distributions across settings (e.g., Wasserstein or KS).
- Within-fold uncertainty: bootstrap CIs for s_norm and p_norm per fold.

## 6) Experiment plan (MVS first, then bounded stretches)

Minimum viable study (default deliverable):
- Model: XGBoost trees.
- Class ratios: 10%, 30%, 50% positives in resampled training folds.
- Evaluation: repeated nested CV with modest HPO budget.
- Outputs: performance metrics; SHAP and PFI importances per fold; stability and agreement analyses for RQ1-RQ2.

Stretch A - stronger HPO (compute permitting):
- Use SMAC HPO Facade (more trials) or Random Facade with Sobol sampling.

Stretch B - correlated-feature duplication ablation (RQ3, small and contained):
- Duplicate 3-5 features as near-copies with small Gaussian noise.
- Rerun MVS for a single ratio (e.g., 30%).
- Report split vs combined importance and change relative to baseline.
- Optional add-on: grouped importance for original + duplicate pairs.

Stretch C - linear baseline (robustness):
- Add logistic regression or gblinear on the same MVS grid if feasible.

Novelty stretch D - PFI metric sensitivity under imbalance (bounded):
- Recompute PFI using 2-3 metrics (ROC AUC, PR AUC, log-loss).
- Output how stability and agreement change with metric choice.

Novelty stretch E - within-fold uncertainty (bounded):
- Bootstrap the test fold to obtain CIs for the global importance vector.
- Separate within-fold estimator noise from across-fold variability.

Novelty stretch F - direction and distribution (bounded, top features only):
- For top-k features, compute mean signed SHAP and simple distributional distances.

## 7) Feedback-session checkpoints

Checkpoint 1 (early, before long runs): scope + protocol sign-off
- Confirm MVS scope is acceptable.
- Walk through nested CV pseudocode and confirm it matches expectations.
- Confirm uncertainty summaries (outer repeats vs single outer split).
- Confirm HPO budget and metric.

Checkpoint 2 (after first results): results sanity check
- Validate trends and confidence intervals.
- Confirm agreement metrics are interpreted correctly (especially when importances are flat).
- Decide whether stretch goals are worthwhile.

## 8) Work packages and acceptance criteria

WP0 - Prior work verification
- Output: `docs/prior_work_check.md`.
- Done when: scope and claims are updated based on prior work; 1-2 novelty stretches chosen (or rejected).

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
- Done when: summary plots for RQ1-RQ2 are generated.

WP4 - Stretch goals (optional)
- Stretch A: stronger HPO
- Stretch B: duplication ablation
- Stretch C: linear baseline
- Novelty D/E/F: pick at most 1-2
- Done when: each selected stretch has a short summary and plots.

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
- Mitigation: lock MVS first; keep HPO budget small; run stretch goals only if MVS is complete.

Risk: Validation protocol is misunderstood or leaky.
- Mitigation: WP2 protocol doc + sanity tests; review in checkpoint 1.

Risk: Rank metrics are noisy when importances are flat.
- Mitigation: always report magnitude-aware metrics and dispersion.

Risk: Permutation importance variance is high.
- Mitigation: multiple permutation repeats; report uncertainty across folds; optional within-fold bootstrap.

Risk: Novelty stretches expand scope too much.
- Mitigation: pick at most 1-2 novelty stretches, each bounded to reuse MVS runs or limited to top-k features.
