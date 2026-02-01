# Prior work reading list (credit scoring, imbalance, SHAP, PFI)

This reading list focuses on (1) explanation stability under class imbalance in credit scoring,
(2) SHAP foundations and tree-specific SHAP used in common credit models, and (3) permutation
feature importance (PFI) pitfalls and fixes (especially under imbalance and feature dependence).

## Reading list (12 sources)

1) Chen, Calabrese & Martin-Barragan (2024). *Interpretable machine learning for imbalanced credit scoring datasets.* European Journal of Operational Research.  
   DOI: 10.1016/j.ejor.2023.06.036  
   Link: https://doi.org/10.1016/j.ejor.2023.06.036  
   Notes: Credit-scoring-focused study showing how increasing class imbalance can reduce the stability of local/global explanations (incl. SHAP-style attributions), with multiple stability indices and controlled imbalance settings.

2) Ballegeer, Bogaert & Benoit (2025). *Evaluating the stability of model explanations in instance-dependent cost-sensitive credit scoring.* European Journal of Operational Research.  
   DOI: 10.1016/j.ejor.2025.05.039  
   Link: https://doi.org/10.1016/j.ejor.2025.05.039  
   Notes: Connects explanation stability to cost-sensitive credit scoring objectives; useful for arguing that stability is affected by both data properties (imbalance) and training objectives.

3) Lin & Wang (2025). *SHAP Stability in Credit Risk Management: A Case Study in Credit Card Default Model.* Risks.  
   DOI: 10.3390/risks13120238  
   Link: https://doi.org/10.3390/risks13120238  
   Notes: Case study focusing on the variability of SHAP-based conclusions (e.g., across runs/splits) in a credit-default context; motivates uncertainty reporting for "global" SHAP importance.

4) Alonso & Carbo (2022). *Accuracy of explanations of machine learning models for credit decisions.* Banco de Espana Working Paper No. 2222.  
   DOI: 10.2139/ssrn.4144780  
   Link: https://doi.org/10.2139/ssrn.4144780  
   Notes: Evaluates how well popular explanation methods match known/constructed ground truth in credit decision settings; helpful for framing evaluation beyond plausibility.

5) Bucker, Szepannek, Gosiewska & Biecek (2022). *Transparency, auditability, and explainability of machine learning models in credit scoring.* Journal of the Operational Research Society.  
   DOI: 10.1080/01605682.2021.1922098  
   Link: https://doi.org/10.1080/01605682.2021.1922098  
   Notes: Credit-scoring-specific framework for explainability and auditability; useful for situating SHAP/PFI within governance, validation, and stakeholder communication.

6) Bussmann, Giudici, Marinelli & Papenbrock (2021). *Explainable Machine Learning in Credit Risk Management.* Computational Economics.  
   DOI: 10.1007/s10614-020-10042-0  
   Link: https://doi.org/10.1007/s10614-020-10042-0  
   Notes: Applied paper in credit risk management; includes model/explanation workflows and discussion of practical considerations for explainable credit models.

7) Visani et al. (2022). *Statistical stability indices for LIME: Obtaining reliable explanations for machine learning models.* Journal of the Operational Research Society.  
   DOI: 10.1080/01605682.2020.1865846  
   Link: https://doi.org/10.1080/01605682.2020.1865846  
   Notes: Defines stability indices and an evaluation approach for explanation reliability on tabular data; commonly used as a methodological backbone for later "stability in credit scoring" studies.

8) Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS (arXiv).  
   DOI: 10.48550/arXiv.1705.07874  
   Link: https://doi.org/10.48550/arXiv.1705.07874  
   Notes: Foundational SHAP paper--axioms and additive feature attribution framework; essential whenever SHAP values are used.

9) Lundberg et al. (2020). *From Local Explanations to Global Understanding with Explainable AI for Trees.* Nature Machine Intelligence.  
   DOI: 10.1038/s42256-019-0138-9  
   Link: https://doi.org/10.1038/s42256-019-0138-9  
   Notes: Core reference for TreeSHAP and aggregating local SHAP to global patterns, highly relevant for tree/boosting models common in credit scoring.

10) Janitza, Strobl & Boulesteix (2013). *An AUC-based permutation variable importance measure for random forests.* BMC Bioinformatics.  
    DOI: 10.1186/1471-2105-14-119  
    Link: https://doi.org/10.1186/1471-2105-14-119  
    Notes: Important for imbalance: shows how permutation importance can be distorted as class imbalance grows, and motivates AUC-based permutation importance for classification.

11) Strobl et al. (2008). *Conditional variable importance for random forests.* BMC Bioinformatics.  
    DOI: 10.1186/1471-2105-9-307  
    Link: https://doi.org/10.1186/1471-2105-9-307  
    Notes: Classic reference on correlation bias in permutation importance; conditional permutations reduce bias when predictors are correlated (common in credit features).

12) Hooker, Mentch & Zhou (2021). *Unrestricted permutation forces extrapolation: variable importance requires at least one more model, or there is no free variable importance.* Statistics and Computing.  
    DOI: 10.1007/s11222-021-10057-z  
    Link: https://doi.org/10.1007/s11222-021-10057-z  
    Notes: Key warning for PFI: naive permutations can create unrealistic (out-of-distribution) feature combinations under dependence, producing misleading importance scores; useful for interpretation guardrails in credit scoring.
