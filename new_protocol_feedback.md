Yes — this protocol is materially better, and it fixes the main “rigor/documentation mismatch” problem from before. It now describes a leakage-safe nested CV procedure in a way that matches the intent of the new harness: one-hot encoding is fold-local, the outer test fold is never resampled, and inner-CV resampling happens only on the inner training split (so duplicates can’t leak into inner validation). 

A few things in here are especially strong from a statistical-rigor standpoint:

* You explicitly state that categorical one-hot encoding is done “within each fold” (fit on training, applied to test with aligned columns). That removes an easy leakage objection. 
* You distinguish outer vs inner responsibilities and state the key safety rule: “inner training fold is resampled… inner test fold is encoded to the same columns.” That directly addresses the duplicate-across-inner-splits risk that happens if you resample first and then split. 
* You specify that PFI uses the primary metric for scoring, which is important for interpretability and consistency of comparisons. 
* The “Sanity checks” section is compact but covers the right invariants (no train/test overlap, determinism, NaNs in single-class folds). 

The only meaningful “still needs tightening” issue is a small inconsistency between the pseudocode and the detailed inner-CV description:

* In the pseudocode, the line “resample train fold to ratio” appears before “run inner CV to select hyperparameters,” which reads like inner CV is run on the already-resampled outer training fold. But your “Inner CV” section correctly states the inner CV splits the original outer training fold and resamples only the inner training partition per split. 

That’s easy to fix by adjusting the pseudocode so it matches the text. For example (same idea, but unambiguous):

```text
for each outer split:
  split X_outer_train, X_outer_test
  for each ratio:
    if HPO enabled:
      run inner CV on X_outer_train:
        for each inner split:
          resample only X_inner_train to ratio
          encode inner train; encode inner val to same columns
          fit/evaluate; aggregate inner metric
      choose best params
    resample full X_outer_train to ratio
    encode outer train; encode outer test to same columns
    fit on outer train; evaluate on untouched outer test
    compute SHAP + PFI on outer test
```

Two additional micro-improvements that would make this read “reviewer-proof,” without expanding it much:

1. Terminology: in the inner CV section, consider calling it “inner validation fold” rather than “inner test fold,” to avoid confusion with the outer test fold (you already use “test fold” for the outer split above). 

2. Randomness/seed scope: you already mention determinism, but if you add one sentence like “All stochastic components (outer split, inner split, resampling, PFI permutations) are seeded from a base seed deterministically as a function of (repeat, fold, ratio),” you’ll eliminate ambiguity about reproducibility. 

So overall: yes, this is better, and it’s better in the exact way that matters statistically. With the pseudocode adjusted to match the inner-CV bullet list (and optionally the small terminology/seed clarifications), it will be a clean, defensible description of a leakage-safe evaluation protocol.

