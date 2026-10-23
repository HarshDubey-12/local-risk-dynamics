"""The improved modelling pipeline (in development).

Planned design (see docs/PIPELINE.md and docs/MODEL_ASSESSMENT.md):

1. Data validation and analysis report, run once (in ``lrd.core``).
2. One shared chronological split / set of walk-forward folds (``lrd.core``).
3. Per model, fitted on training data only: model-specific preparation
   (e.g. standardised factors), hyperparameter tuning, fit and predict.
4. Shared evaluation and significance tests (``lrd.core.evaluation``).

First model: MCLLR with local ridge shrinkage, standardised kernel distances
and predictive variance (validated in scripts/improvement_study.py).
"""
