"""The improved modelling system (see docs/PIPELINE.md).

1. Raw data is validated once (``lrd.core.data.validation``).
2. Every model is evaluated on the same walk-forward folds (``lrd.core``).
3. Inside each fold, on training data only: model-specific preparation
   (``preprocessing.PreparedModel``) and nested hyperparameter tuning
   (``tuning.TunedModel``), then fit and predict.
4. Shared metrics, calibration and significance tests against a benchmark
   (``pipeline.ImprovedPipeline``).

Models: ``HistoricalMean`` (benchmark), ``RidgeRegression`` and
``ImprovedMCLLR`` (local ridge, standardised distances, predictive variance).
"""
