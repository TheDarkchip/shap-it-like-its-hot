# Prior-work check

Date: 2026-01-21

This document checks whether prior work already answers the project questions around (i) explanation stability under class imbalance and (ii) SHAP vs PFI agreement in credit-scoring-style tabular data. It also identifies bounded novelty stretches for a compact replication/extension.

The outcome of this check is reflected in `ROADMAP.md` (updated research questions, metrics, and scope).

## 1) Executive summary (TL;DR)

- Class imbalance and explanation stability in credit scoring is already active, partially answered work. This repo should frame RQ1 as replication/extension with clear added value (PFI under the same protocol, uncertainty reporting).
- PFI can degrade under imbalance depending on the metric; AUC-based scoring is defensible by default. Metric sensitivity is a credible novelty stretch.
- SHAP vs PFI disagreement on German Credit already exists in prior work; the repo should focus on a careful protocol (repeated nested CV, untouched test folds) and magnitude-aware interpretation.
- Disagreement framing is established; novelty should come from experimental control and reporting rather than new metrics.
- Repeated nested CV explanation aggregation has prior guidance; follow a defensible protocol and be explicit about deviations.
- Correlation confounds both permutation FI and SHAP; treat correlated-feature ablations as sanity checks, not novelty claims.

## 2) Coverage map: RQ vs prior work vs gap

| Research question | Closest prior work | What is already answered | Remaining gap for this repo |
| --- | --- | --- | --- |
| RQ1: Stability under class imbalance | Chen et al. 2024; Ballegeer et al. 2025 | Imbalance reduces stability (SHAP/LIME) in credit scoring | Replicate stability trend with repeated nested CV and add PFI stability under the same protocol |
| RQ2: SHAP vs PFI agreement under imbalance | Markus et al. 2023; Krishna et al. 2022 | Disagreement is common; German Credit included | Agreement under controlled training imbalance with magnitude-aware reporting and uncertainty |
| RQ3: Correlated features (optional) | Strobl et al. 2008; Hooker et al. 2021; Aas et al. 2021 | Correlation confounds both PFI and SHAP | Small near-duplicate ablation as an interpretation aid; optional grouped importance |

## 3) Key implications for scope

- Treat RQ1 as replication/extension: add PFI stability and uncertainty summaries under the same protocol.
- Avoid claiming novelty for SHAP vs PFI disagreement on German Credit; focus on protocol rigor and interpretation guardrails.
- Use AUC-based PFI as the default; consider a small metric-sensitivity stretch.
- Correlated-feature tests should be bounded and labeled as sanity checks.

## 4) Scope decisions adopted in ROADMAP.md

- Dataset: Statlog German Credit (single dataset).
- Primary model: XGBoost trees (TreeSHAP); optional linear baseline.
- Evaluation: repeated nested CV; explanations computed on untouched outer test folds.
- Class ratio manipulation: resampling only inside outer training folds.
- Metrics: rank + magnitude stability and agreement, with importance dispersion guardrails.

## 5) Novelty stretch candidates (bounded)

Low-cost, high-value:
1) PFI metric sensitivity under imbalance
- Recompute PFI using 2-3 metrics (ROC AUC, PR AUC, log-loss) on the same trained models/splits.
- Report how stability (RQ1) and agreement (RQ2) depend on the metric choice.

2) Directionality and sign agreement for top features
- Report mean signed SHAP for top-k features alongside mean(|SHAP|).
- Track sign agreement and signed-rank agreement across folds.

Medium-cost, still bounded:
3) Within-fold uncertainty for global importance estimates
- Bootstrap the test fold to get CIs for the global importance vector.
- Separate within-fold estimator noise from across-fold variability.

4) Grouped importance for correlated pairs
- Evaluate whether grouping original + duplicate features improves agreement.

Higher-cost (only if time permits):
5) Dependence-aware PFI (conditional permutation) as a comparison point
- Compare unconditional vs conditional PFI on the correlated-feature ablation.
- Treat as illustrative only.

## 6) Updated research questions (used going forward)

RQ1 (main) - Class imbalance vs stability (global explanations)
How does varying the training class ratio (10%, 30%, 50% positives), while evaluating on untouched held-out folds, affect the stability of global SHAP and PFI importances (rank stability + magnitude stability)?

RQ2 (main) - SHAP vs PFI agreement under imbalance (rank + magnitude)
Across the same imbalance settings, when do SHAP and PFI agree or disagree on important features, and is disagreement primarily about ranking or magnitude (especially in near-flat importance regimes)?

RQ3 (optional) - Correlated features
When a small number of near-duplicate features are introduced, how do SHAP and PFI allocate importance across correlated pairs, and does grouping correlated features improve apparent agreement or stability?

## 7) References (key items)

- Chen, Y., Calabrese, R., and Martin-Barragan, B. Interpretable machine learning for imbalanced credit scoring datasets. European Journal of Operational Research (2024). DOI: 10.1016/j.ejor.2023.06.036
- Ballegeer, M., Bogaert, M., and Benoit, D. F. Evaluating the stability of model explanations in instance-dependent cost-sensitive credit scoring. European Journal of Operational Research (2025). DOI: 10.1016/j.ejor.2025.05.039
- Markus, A. F., Fridgeirsson, E. A., Kors, J. A., Verhamme, K. M. C., Reps, J. M., and Rijnbeek, P. R. Understanding the Size of the Feature Importance Disagreement Problem in Real-World Data. IMLH workshop at ICML (2023). OpenReview: https://openreview.net/pdf?id=FKjFUEV63f
- Krishna, S., Han, T., Gu, A., Wu, S., Jabbari, S., and Lakkaraju, H. The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective. arXiv (2022). DOI: 10.48550/arXiv.2202.01602
- Janitza, S., Strobl, C., and Boulesteix, A.-L. An AUC-based permutation variable importance measure for random forests. BMC Bioinformatics (2013). DOI: 10.1186/1471-2105-14-119
- Scheda, R., and Diciotti, S. Explanations of Machine Learning Models in Repeated Nested Cross-Validation: An Application in Age Prediction Using Brain Complexity Features. Applied Sciences (2022). DOI: 10.3390/app12136681
- Strobl, C., Boulesteix, A.-L., Kneib, T., Augustin, T., and Zeileis, A. Conditional variable importance for random forests. BMC Bioinformatics (2008). DOI: 10.1186/1471-2105-9-307
- Hooker, G., Mentch, L., and Zhou, S. Unrestricted permutation forces extrapolation: variable importance requires at least one more model, or there is no free variable importance. Statistics and Computing (2021). DOI: 10.1007/s11222-021-10057-z
- Aas, K., Jullum, M., and Loland, A. Explaining individual predictions when features are dependent: More accurate approximations to Shapley values. Artificial Intelligence (2021). DOI: 10.1016/j.artint.2021.103502
- Lin, L., and Wang, Y. SHAP Stability in Credit Risk Management: A Case Study in Credit Card Default Model. Risks (2025). DOI: 10.3390/risks13120238
- Alonso, A., and Carbo, J. M. Accuracy of explanations of machine learning models for credit decisions. Banco de Espana Working Paper No. 2222 (2022). SSRN DOI: 10.2139/ssrn.4144780
