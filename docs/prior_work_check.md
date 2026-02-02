# Prior-work check

Last updated: 2026-02-01

This document checks whether prior work already answers the project questions around (i) explanation stability under class imbalance and (ii) SHAP vs PFI agreement in credit-scoring-style tabular data. It also locks scope and selects bounded novelty stretches for a compact replication/extension.

Companion doc (per-source summaries + RQ mapping): `docs/prior_work_summaries.md`.

## 1) Executive summary (TL;DR)

- Class imbalance and explanation stability in credit scoring is already active, partially answered work. This repo should frame RQ1 as replication/extension with clear added value (PFI under the same protocol, uncertainty reporting).
- PFI can degrade under imbalance depending on the scoring metric; AUC-based scoring is defensible by default. Metric sensitivity is a credible novelty stretch that remains bounded.
- SHAP vs PFI disagreement on German Credit (and similar tabular settings) already exists in prior work; the repo should focus on a careful protocol (repeated nested CV, untouched test folds) and magnitude-aware interpretation (not just rank correlations).
- Disagreement framing is established; novelty should come from experimental control and reporting guardrails rather than proposing new disagreement metrics.
- Repeated nested CV explanation aggregation has prior guidance; follow a defensible protocol and be explicit about deviations.
- Correlation confounds both permutation FI and SHAP; treat correlated-feature ablations as sanity checks, not headline novelty claims.

## 2) Coverage map: RQ vs prior work vs gap

| Research question | Closest prior work | What is already answered | Remaining gap for this repo |
| --- | --- | --- | --- |
| RQ1: Stability under class imbalance | Chen et al. 2024; Ballegeer et al. 2025 | Imbalance reduces explanation stability (shown for SHAP/LIME) in credit scoring | Replicate the stability trend with repeated nested CV and add **PFI stability** under the same protocol |
| RQ2: SHAP vs PFI agreement under imbalance | Markus et al. 2023; Krishna et al. 2022 | Disagreement is common; German Credit included | Agreement under controlled training imbalance with magnitude-aware reporting and uncertainty |
| RQ3: Correlated features (optional) | Strobl et al. 2008; Hooker et al. 2021; Aas et al. 2021 | Correlation confounds both PFI and SHAP | Small near-duplicate ablation as an interpretation aid; optional grouped importance |

## 3) Annotated core sources (8–12)

These are the most load-bearing sources for the repo’s claims and scope. Shorter per-source summaries + explicit RQ mapping live in `docs/prior_work_summaries.md`.

1) Chen, Calabrese & Martin-Barragán (2024). Interpretable machine learning for imbalanced credit scoring datasets. *EJOR.* DOI: 10.1016/j.ejor.2023.06.036  
Main takeaway: in credit scoring, explanation stability (incl. SHAP) decreases as class imbalance increases; this is closest overlap for RQ1 and drives the “replicate + extend with PFI” framing.

2) Ballegeer, Bogaert & Benoit (2025). Evaluating the stability of model explanations in instance-dependent cost-sensitive credit scoring. *EJOR.* DOI: 10.1016/j.ejor.2025.05.039  
Main takeaway: extends stability-under-imbalance work to cost-sensitive objectives; supports treating stability as a risk-management metric (RQ1 motivation).

3) Markus et al. (2023). Understanding the Size of the Feature Importance Disagreement Problem in Real-World Data. *IMLH@ICML.* OpenReview: https://openreview.net/forum?id=FKjFUEV63f  
Main takeaway: SHAP vs PFI (and other FI methods) disagreement is empirically large; includes German Credit and shows disagreement depends on data properties (RQ2 framing; partial RQ3 relevance).

4) Krishna et al. (2022). The Disagreement Problem in Explainable Machine Learning: A Practitioner’s Perspective. arXiv. DOI: 10.48550/arXiv.2202.01602  
Main takeaway: establishes “disagreement” as a known phenomenon (not a novelty claim) and motivates measuring agreement explicitly (RQ2).

5) Janitza, Strobl & Boulesteix (2013). An AUC-based permutation variable importance measure for random forests. *BMC Bioinformatics.* DOI: 10.1186/1471-2105-14-119  
Main takeaway: permutation importance can behave poorly under imbalance if paired with unsuitable metrics; supports AUC-based PFI defaults and motivates metric-sensitivity stretch.

6) Strobl et al. (2008). Conditional variable importance for random forests. *BMC Bioinformatics.* DOI: 10.1186/1471-2105-9-307  
Main takeaway: permutation importance is biased under correlated predictors; conditional schemes mitigate this (core for RQ3 and for PFI interpretation caveats).

7) Hooker, Mentch & Zhou (2021). Unrestricted permutation forces extrapolation. *Statistics and Computing.* DOI: 10.1007/s11222-021-10057-z  
Main takeaway: naive permutation can create out-of-distribution samples when features are dependent; important guardrail for PFI in correlated-feature settings (RQ3).

8) Aas, Jullum & Løland (2021). Explaining individual predictions when features are dependent. *Artificial Intelligence.* DOI: 10.1016/j.artint.2021.103502  
Main takeaway: SHAP approximations can misbehave under dependence; dependence-aware Shapley approximations and grouping ideas are relevant for RQ3 interpretation.

9) Visani et al. (2022). Statistical stability indices for LIME. *JORS.* DOI: 10.1080/01605682.2020.1865846  
Main takeaway: provides stability-index vocabulary and a credit-risk case study; supports the general idea that explainability requires stability evaluation (RQ1 framing).

10) Scheda & Diciotti (2022). Explanations in repeated nested CV. *Applied Sciences.* DOI: 10.3390/app12136681  
Main takeaway: protocol guidance for explanation computation/aggregation in repeated nested CV (supports your planned evaluation design).

11) Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. arXiv/NeurIPS. DOI: 10.48550/arXiv.1705.07874  
Main takeaway: foundational SHAP axioms/definition; required background for any SHAP-based claim.

12) Lundberg et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence.* DOI: 10.1038/s42256-019-0138-9  
Main takeaway: TreeSHAP and local-to-global aggregation for tree ensembles (the repo’s primary modeling choice).

## 4) Final scope decisions (confirmed)

Locked scope (minimum viable study):
- Dataset: Statlog German Credit (single dataset).
- Primary model: XGBoost trees with TreeSHAP.
- Evaluation: repeated nested CV; explanations computed on untouched outer test folds (no leakage).
- Class-ratio manipulation: resampling only inside each outer training fold to target 10%, 30%, 50% positives (test folds untouched).
- Global importance definitions:
  - SHAP: mean(|SHAP|) on each outer test fold
  - PFI: permutation importance on each outer test fold with multiple repeats
- Primary reporting:
  - Stability within method: rank + magnitude stability, with “flat-importance” guardrails
  - Agreement between methods: rank + magnitude agreement, plus top-k overlap

Non-goals for this repo (unless explicitly expanded later):
- multiple datasets, many additional explainers, large model families, causal claims about “true importance”.

## 5) Novelty stretches: selected vs deferred

Selected (bounded; do after MVS is complete):
1) PFI metric sensitivity under imbalance (novelty stretch D)
- Recompute PFI using 2–3 scoring metrics (ROC AUC, PR AUC, log-loss) on the same trained models/splits.
- Report how stability (RQ1) and agreement (RQ2) depend on metric choice.

2) Directionality / sign agreement for SHAP (novelty stretch F, lightweight)
- Report mean signed SHAP for top-k features alongside mean(|SHAP|).
- Track sign agreement (and optionally signed-rank agreement) across folds and imbalance settings.

Deferred / optional (bounded but not committed):
- Within-fold bootstrap CIs for global importance vectors (extra compute; useful but not required for the core story).
- Correlated-feature duplication ablation (RQ3): keep as a sanity check if time permits; avoid novelty claims.

## 6) Research questions (locked)

RQ1 (main) – Class imbalance vs stability (global explanations)  
How does varying the training class ratio (10%, 30%, 50% positives), while evaluating on untouched held-out folds, affect the stability of global SHAP and PFI importances (rank stability + magnitude stability)?

RQ2 (main) – SHAP vs PFI agreement under imbalance (rank + magnitude)  
Across the same imbalance settings, when do SHAP and PFI agree or disagree on important features, and is disagreement primarily about ranking or magnitude (especially in near-flat importance regimes)?

RQ3 (optional) – Correlated features sanity check  
When a small number of near-duplicate features are introduced, how do SHAP and PFI allocate importance across correlated pairs, and does grouping correlated features improve apparent agreement or stability?

## 7) Additional credit-scoring XAI references (useful context)

These are not the core overlap papers for RQ1–RQ3, but help for motivation and reporting in a credit-risk context:
- Bücker et al. (2022). Transparency, auditability, and explainability of ML models in credit scoring. *JORS.* DOI: 10.1080/01605682.2021.1922098
- Bussmann et al. (2021). Explainable Machine Learning in Credit Risk Management. *Computational Economics.* DOI: 10.1007/s10614-020-10042-0
- Alonso & Carbó (2022). Accuracy of explanations of ML models for credit decisions. Banco de España WP 2222. SSRN DOI: 10.2139/ssrn.4144780
- Lin & Wang (2025). SHAP Stability in Credit Risk Management: Credit Card Default case study. *Risks.* DOI: 10.3390/risks13120238
