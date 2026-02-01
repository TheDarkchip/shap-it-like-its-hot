# Prior work summaries + RQ coverage map (credit scoring: imbalance, SHAP, PFI)

Generated: 2026-02-01

This note provides short, per-source summaries and a quick mapping to the repository research questions (RQ1–RQ3). RQ wording matches `docs/prior_work_check.md` and `ROADMAP.md`.

## Research questions (repo wording)

RQ1 (stability under class imbalance, global explanations): How does varying the training class ratio (e.g., 10%, 30%, 50% positives), while evaluating on untouched held-out folds, affect the stability of global SHAP and PFI importances (rank stability + magnitude stability)?

RQ2 (SHAP vs PFI agreement under imbalance): Across the same imbalance settings, when do SHAP and PFI agree or disagree on important features, and is disagreement primarily about ranking or magnitude (especially in near-flat importance regimes)?

RQ3 (optional sanity check: correlated features): When near-duplicate/correlated features are introduced, how do SHAP and PFI allocate importance across correlated pairs, and does grouping correlated features improve apparent agreement or stability?

## Quick coverage matrix

Legend:
- **P** = primary / directly addresses the RQ
- **S** = supporting / background / partial relevance
- **–** = not addressed

| Source | What it’s mainly about | RQ1 | RQ2 | RQ3 | How it’s useful in this repo |
| --- | --- | --- | --- | --- | --- |
| Chen et al. (2024, EJOR) | Imbalance → instability of SHAP/LIME in credit scoring | **P** | – | – | Closest overlap for RQ1; motivates “replicate + add PFI” |
| Ballegeer et al. (2025, EJOR) | Cost-sensitive credit scoring + stability of SHAP/LIME under imbalance | **P** | – | – | RQ1 with objective-function twist (cost efficiency vs stability) |
| Lin & Wang (2025, Risks / arXiv) | SHAP stability across repeated model fits in credit default | **S** | – | – | Practical “SHAP variability exists” evidence + reporting guidance |
| Visani et al. (2022, JORS) | Stability indices for LIME + credit-risk case study | **S** | – | – | Stability-metric vocabulary (VSI/CSI-style thinking) |
| Scheda & Diciotti (2022, Applied Sciences) | How to compute/aggregate SHAP inside repeated nested CV | **S** | **S** | – | Protocol justification: compute explanations on untouched test folds |
| Markus et al. (2023, IMLH@ICML) | Why FI methods disagree; role of data “complexity” incl. correlation | – | **P** | **S** | Direct prior on disagreement + includes German Credit |
| Krishna et al. (2022, arXiv) | “Disagreement problem” framing + empirical disagreement study | – | **P** | – | Canonical citation for why SHAP vs PFI can conflict |
| Lundberg & Lee (2017, arXiv/NeurIPS) | SHAP foundations + axioms | **S** | **S** | **S** | Required background for any SHAP usage/claims |
| Lundberg et al. (2020, Nat Mach Intell) | TreeSHAP + local-to-global aggregation for trees | **S** | **S** | **S** | Justifies TreeSHAP and global importance via aggregation |
| Janitza et al. (2013, BMC Bioinf) | Permutation importance under imbalance; AUC-based PFI | **S** | **S** | – | Supports “PFI metric choice matters under imbalance” |
| Strobl et al. (2008, BMC Bioinf) | Conditional permutation importance for correlated predictors | **S** | **S** | **P** | Core citation for correlation bias in permutation importance |
| Hooker et al. (2021, Stat & Comp) | Permutation can force extrapolation (OOD samples) under dependence | **S** | **S** | **P** | Theory/intuition for why naive PFI can mislead with correlation |
| Aas et al. (2021, Artificial Intelligence) | Shapley values with dependent features; grouping dependent vars | – | **S** | **P** | Core citation for dependence-aware SHAP variants and grouping |
| Bücker et al. (2022, JORS) | Transparency/auditability/explainability in credit scoring | **S** | **S** | **S** | Positioning: how to present/validate explanations in credit scoring |
| Bussmann et al. (2021, Computational Economics) | Shapley values in credit risk + grouping via correlation networks | **S** | **S** | **S** | Credit-risk exemplar for clustering/grouping SHAP explanations |
| Alonso & Carbó (2022, Banco de España WP) | “Accuracy” evaluation of explanation methods using synthetic DGPs | **S** | **S** | – | Evaluation framing: explanation quality is testable, not assumed |

## Per-source summaries + RQ mapping

### 1) Chen, Y., Calabrese, R., & Martin-Barragán, B. (2024). *Interpretable machine learning for imbalanced credit scoring datasets.* European Journal of Operational Research.
DOI: https://doi.org/10.1016/j.ejor.2023.06.036  
Link: https://www.sciencedirect.com/science/article/pii/S0377221723005088

Summary: This paper is the closest match to RQ1. It proposes a controlled resampling setup that creates multiple versions of a credit dataset with progressively higher class imbalance (lower default rate), while keeping sample sizes comparable. Using credit scoring data (UK residential mortgages, plus open-source credit scoring datasets for robustness), it studies how explanation stability changes for LIME and SHAP when the training data becomes more imbalanced. Stability is evaluated both as ranking stability (e.g., sequential rank agreement) and magnitude stability (e.g., variation in attribution magnitudes), and the headline finding is that explanations become less stable as imbalance increases.

RQ mapping: RQ1 = **Primary** (directly studies imbalance → stability); RQ2 = –; RQ3 = –  
How to use it: (i) cite as “imbalance hurts explanation stability in credit scoring”; (ii) frame this repo as a replication/extension by adding PFI under the same protocol.

---

### 2) Ballegeer, M., Bogaert, M., & Benoit, D. F. (2025). *Evaluating the stability of model explanations in instance-dependent cost-sensitive credit scoring.* European Journal of Operational Research.
DOI: https://doi.org/10.1016/j.ejor.2025.05.039  
Link: https://www.sciencedirect.com/science/article/abs/pii/S0377221725004230  
Preprint (accepted manuscript): https://arxiv.org/pdf/2509.01409

Summary: Extends the “imbalance → instability” story to a cost-sensitive setting. The paper studies instance-dependent cost-sensitive (IDCS) classifiers for credit scoring and then evaluates explanation stability (SHAP and LIME) under different imbalance levels created via controlled resampling. The main contribution, relative to Chen et al., is that it emphasizes a trade-off: cost-sensitive models can improve cost efficiency, but their post-hoc explanations can be substantially less stable—especially as class imbalance grows. It reinforces the idea that explanation stability is a model-risk concern, not just a methodological nicety.

RQ mapping: RQ1 = **Primary**; RQ2 = –; RQ3 = –  
How to use it: cite as motivation for treating stability as a first-class risk-management metric; helpful when discussing “objective choice can affect stability.”

---

### 3) Lin, L., & Wang, Y. (2025). *SHAP Stability in Credit Risk Management: A Case Study in Credit Card Default Model.* Risks.
DOI: https://doi.org/10.3390/risks13120238  
Link: https://www.mdpi.com/2227-9091/13/12/238  
arXiv: https://arxiv.org/abs/2508.01851

Summary: Focuses on SHAP stability across repeated model training runs (e.g., different random seeds) on a credit card default dataset. The key practical message is that SHAP-based feature importance can vary across “equally reasonable” model fits, and that variability depends on the importance level (top features tend to be more consistent than borderline features). While this is not an imbalance manipulation study, it is a useful credit-risk-specific reference for motivating repeated runs and uncertainty reporting for SHAP importances.

RQ mapping: RQ1 = **Supporting** (stability exists and should be measured, but not primarily imbalance-driven); RQ2 = –; RQ3 = –  
How to use it: cite when justifying repeated outer CV, multiple seeds, and confidence intervals for global SHAP.

---

### 4) Visani, G., Bagli, E., Chesani, F., Poluzzi, A., & Capuzzo, D. (2022). *Statistical stability indices for LIME: obtaining reliable explanations for machine learning models.* Journal of the Operational Research Society.
DOI: https://doi.org/10.1080/01605682.2020.1865846  
Link: https://www.tandfonline.com/doi/abs/10.1080/01605682.2020.1865846

Summary: Introduces complementary stability indices for LIME (often referred to via feature-set stability and coefficient stability ideas) and argues that stability assessment is essential if LIME explanations are to be trusted. The paper includes a credit risk case study comparing ML and classical statistical techniques and demonstrates how the proposed indices can surface instability in explanation outputs. Even if your repo doesn’t use LIME directly, this paper helps anchor the broader notion that “explainability must be evaluated” and provides historical context for stability metrics used in later credit-scoring stability papers.

RQ mapping: RQ1 = **Supporting**; RQ2 = –; RQ3 = –  
How to use it: cite for stability-index terminology and for the general claim that explanation reliability requires stability assessment.

---

### 5) Scheda, R., & Diciotti, S. (2022). *Explanations of Machine Learning Models in Repeated Nested Cross-Validation: An Application in Age Prediction Using Brain Complexity Features.* Applied Sciences.
DOI: https://doi.org/10.3390/app12136681  
Link: https://www.mdpi.com/2076-3417/12/13/6681

Summary: Methodology paper on how to compute representative SHAP values inside a repeated nested cross-validation procedure, and how to keep training vs test explanations separate to assess “generalization of explanations.” Although the application is not credit scoring, it supports the repo’s planned evaluation design: explanations should be computed on untouched test folds (outer CV) to avoid overly optimistic explanation stability/agreement claims. It’s a good protocol citation when writing up “why repeated nested CV, why compute explanations on outer test folds.”

RQ mapping: RQ1 = **Supporting**; RQ2 = **Supporting**; RQ3 = –  
How to use it: cite when defending the experimental protocol (repeatability + avoiding leakage in explanations).

---

### 6) Markus, A. F., Fridgeirsson, E. A., Kors, J. A., Verhamme, K. M. C., Reps, J. M., & Rijnbeek, P. R. (2023). *Understanding the Size of the Feature Importance Disagreement Problem in Real-World Data.* IMLH workshop at ICML (OpenReview).
Link: https://openreview.net/forum?id=FKjFUEV63f (PDF available via OpenReview)

Summary: Addresses the “feature importance disagreement” phenomenon: different global FI methods can yield conflicting rankings/importance vectors. The paper proposes an evaluation framework that perturbs real-world datasets to vary elements of “data complexity” and then measures how disagreement changes. Importantly for this repo, they explicitly apply the framework to German Credit (and COMPAS) and discuss how factors like feature correlation can increase disagreement or spread importance. This is a very direct precedent for RQ2, and also motivates RQ3 as an optional stress test.

RQ mapping: RQ1 = –; RQ2 = **Primary**; RQ3 = **Supporting**  
How to use it: cite as prior evidence that FI disagreement is real on German Credit; position your contribution as “controlled imbalance + nested CV + magnitude-aware reporting.”

---

### 7) Krishna, S., Han, T., Gu, A., Wu, S., Jabbari, S., & Lakkaraju, H. (2022). *The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective.* arXiv.
DOI: https://doi.org/10.48550/arXiv.2202.01602  
arXiv: https://arxiv.org/abs/2202.01602

Summary: A foundational “disagreement” paper that (i) formalizes what it means for explanations to disagree, (ii) empirically measures disagreement across multiple explanation methods, models, and datasets, and (iii) adds a practitioner angle via interviews and a user study. The main takeaway is that explanation methods frequently disagree, and practitioners often resolve conflicts using ad hoc heuristics—raising the risk of relying on misleading explanations in high-stakes settings. While not credit-scoring-specific, it is a canonical reference to justify RQ2.

RQ mapping: RQ1 = –; RQ2 = **Primary**; RQ3 = –  
How to use it: cite as the general framing and motivation for measuring SHAP vs PFI agreement (and not assuming agreement).

---

### 8) Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* arXiv / NeurIPS.
DOI: https://doi.org/10.48550/arXiv.1705.07874  
arXiv: https://arxiv.org/abs/1705.07874

Summary: The core SHAP reference. It frames feature attributions as additive feature attribution methods and shows that Shapley values yield a unique solution under desirable axioms (often cited as local accuracy, missingness, consistency). It also provides practical estimation approaches (e.g., KernelSHAP) and positions SHAP as a unifying lens across multiple explanation techniques. In your repo, this is the “what SHAP is” citation, rather than a direct answer to imbalance/agreement questions.

RQ mapping: RQ1 = **Supporting**; RQ2 = **Supporting**; RQ3 = **Supporting**  
How to use it: cite in Methods when introducing SHAP; cite in Limitations if you discuss assumptions behind SHAP approximations.

---

### 9) Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S.-I. (2020). *From local explanations to global understanding with explainable AI for trees.* Nature Machine Intelligence.
DOI: https://doi.org/10.1038/s42256-019-0138-9  
Link: https://www.nature.com/articles/s42256-019-0138-9

Summary: Key reference for TreeSHAP and for moving from local attributions to global understanding in tree-based models. It provides algorithmic advances for efficient Shapley value computation on trees and introduces analysis tools such as aggregated global patterns and feature interaction effects. Since the repo’s baseline model is tree-based (e.g., XGBoost), this paper supports the choice of TreeSHAP and the practice of aggregating local SHAP values (e.g., mean absolute SHAP) into a global importance vector.

RQ mapping: RQ1 = **Supporting**; RQ2 = **Supporting**; RQ3 = **Supporting**  
How to use it: cite as the SHAP implementation rationale for tree ensembles and for your global aggregation choice.

---

### 10) Janitza, S., Strobl, C., & Boulesteix, A.-L. (2013). *An AUC-based permutation variable importance measure for random forests.* BMC Bioinformatics.
DOI: https://doi.org/10.1186/1471-2105-14-119  
Link: https://link.springer.com/article/10.1186/1471-2105-14-119

Summary: A core citation for doing permutation importance in imbalanced classification. The paper argues that “standard” permutation variable importance paired with misclassification error can behave poorly when the positive class is rare, and proposes an AUC-based permutation importance to mitigate this issue. For this repo, it supports your design choice of using ROC AUC (or similarly imbalance-robust metrics) as the default scoring function for PFI, and it motivates a bounded stretch goal around metric sensitivity (AUC vs PR AUC vs log-loss).

RQ mapping: RQ1 = **Supporting**; RQ2 = **Supporting**; RQ3 = –  
How to use it: cite in Methods when defining PFI scoring; cite in Discussion when explaining why PFI can “look unstable” under imbalance if the metric is poorly chosen.

---

### 11) Strobl, C., Boulesteix, A.-L., Kneib, T., Augustin, T., & Zeileis, A. (2008). *Conditional variable importance for random forests.* BMC Bioinformatics.
DOI: https://doi.org/10.1186/1471-2105-9-307  
Link: https://link.springer.com/article/10.1186/1471-2105-9-307

Summary: Classic paper explaining why permutation importance can be biased in the presence of correlated predictors (and in other situations), and proposing a conditional permutation scheme designed to measure the unique contribution of a variable given correlated alternatives. In credit scoring, where predictors can be correlated (e.g., redundant financial indicators), this is a key citation for any correlation ablation or “grouped importance” interpretation. It is also relevant when interpreting SHAP vs PFI disagreement: some disagreements can be driven by feature dependence rather than model instability.

RQ mapping: RQ1 = **Supporting**; RQ2 = **Supporting**; RQ3 = **Primary**  
How to use it: cite as the methodological basis for your RQ3 (duplicate-feature) sanity check and for cautioning against naïve PFI interpretations under correlation.

---

### 12) Hooker, G., Mentch, L., & Zhou, S. (2021). *Unrestricted permutation forces extrapolation: variable importance requires at least one more model, or there is no free variable importance.* Statistics and Computing.
DOI: https://doi.org/10.1007/s11222-021-10057-z  
Link: https://link.springer.com/article/10.1007/s11222-021-10057-z

Summary: Provides a broad critique of “permute-and-predict” interpretability methods (including permutation importance and partial dependence) when features are dependent. The key argument is that permuting a feature without respecting the joint distribution can create unrealistic feature combinations, effectively forcing the model to extrapolate; the resulting importance can therefore reflect artifacts of out-of-distribution evaluation rather than genuine dependence on that feature in realistic data. This is highly relevant to RQ3 and also serves as a caution in RQ2 (disagreement could be partly due to PFI’s data-manifold violations).

RQ mapping: RQ1 = **Supporting**; RQ2 = **Supporting**; RQ3 = **Primary**  
How to use it: cite in Limitations/Interpretation guardrails for PFI, especially when you discuss correlated-feature tests.

---

### 13) Aas, K., Jullum, M., & Løland, A. (2021). *Explaining individual predictions when features are dependent: More accurate approximations to Shapley values.* Artificial Intelligence.
DOI: https://doi.org/10.1016/j.artint.2021.103502  
Link: https://www.sciencedirect.com/science/article/pii/S0004370221000539  
Open-access PDF: https://martinjullum.com/publication/aas-2021-explaining/aas-2021-explaining.pdf

Summary: Focuses on the fact that many Shapley-value approximations (notably KernelSHAP) assume feature independence, which can generate unrealistic “coalition” samples when features are correlated—leading to misleading explanations even for simple models. The paper proposes extensions that approximate Shapley values while respecting feature dependence (e.g., using conditional distributions, copulas, or empirical approaches) and discusses grouping dependent variables to provide more faithful explanations. While TreeSHAP has its own conventions (interventional vs observational variants), this paper is central to the general message: feature dependence is a first-order issue for attribution methods and can motivate grouping.

RQ mapping: RQ1 = –; RQ2 = **Supporting**; RQ3 = **Primary**  
How to use it: cite as the core dependence-aware SHAP reference; cite if you implement grouped importances or discuss dependence assumptions.

---

### 14) Bücker, M., Szepannek, G., Gosiewska, A., & Biecek, P. (2022). *Transparency, auditability, and explainability of machine learning models in credit scoring.* Journal of the Operational Research Society.
DOI: https://doi.org/10.1080/01605682.2021.1922098  
Link: https://www.tandfonline.com/doi/abs/10.1080/01605682.2021.1922098  
arXiv: https://arxiv.org/abs/2009.13384

Summary: Provides a credit-scoring-specific framework for making ML models transparent and auditable, motivated by regulatory and governance demands. It surveys explainability techniques, discusses how to present them responsibly, and includes a case study showing that “black-box” ML can approach scorecard-like interpretability while improving predictive power. This paper doesn’t directly answer RQ1/RQ2/RQ3, but it is useful framing: it helps justify why explanation stability and method agreement matter in credit risk governance.

RQ mapping: RQ1 = **Supporting**; RQ2 = **Supporting**; RQ3 = **Supporting**  
How to use it: cite in Introduction/Background for regulatory/auditability context and for recommended reporting practices.

---

### 15) Bussmann, N., Giudici, P., Marinelli, D., & Papenbrock, J. (2021). *Explainable Machine Learning in Credit Risk Management.* Computational Economics.
DOI: https://doi.org/10.1007/s10614-020-10042-0  
Link: https://link.springer.com/article/10.1007/s10614-020-10042-0

Summary: An applied credit-risk paper that uses Shapley values and then builds “correlation networks” over explanations to group model predictions by similarity in their explanatory patterns. The contribution is less about stability or imbalance and more about operationalizing SHAP at scale (e.g., clustering borrowers/loans by explanation similarity). It can be used as an example of how SHAP can move beyond per-instance dashboards toward portfolio-level monitoring—a theme that connects naturally to stability and governance, even if not tested under imbalance.

RQ mapping: RQ1 = **Supporting**; RQ2 = **Supporting**; RQ3 = **Supporting**  
How to use it: cite as an example of SHAP usage in credit risk management and for grouping/structuring explanations.

---

### 16) Alonso, A., & Carbó, J. M. (2022). *Accuracy of explanations of machine learning models for credit decisions.* Banco de España Working Paper No. 2222.
SSRN DOI: https://doi.org/10.2139/ssrn.4144780  
Banco de España PDF: https://www.bde.es/f/webbde/SES/Secciones/Publicaciones/PublicacionesSeriadas/DocumentosTrabajo/22/Files/dt2222e.pdf

Summary: Proposes a framework to evaluate how “accurate” explanation methods are by using synthetic datasets where the data-generating process is known, making it possible to define ground-truth feature relevance under controlled assumptions. The main value for this repo is conceptual: it supports the argument that post-hoc explanation methods should be evaluated empirically (and not taken as automatically faithful), and that explanation quality can be studied systematically—an important point when interpreting SHAP/PFI disagreement.

RQ mapping: RQ1 = **Supporting**; RQ2 = **Supporting**; RQ3 = –  
How to use it: cite when motivating evaluation criteria beyond face validity; supports the “treat explanations like statistical estimators with uncertainty” mindset.
