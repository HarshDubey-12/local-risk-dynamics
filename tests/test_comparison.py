"""Cross-system comparison engine tests (lrd.comparison)."""

import numpy as np
import pandas as pd
import pytest

from lrd import comparison as cmp
from lrd.core.data.loader import load_fama_french
from lrd.improved.config import ImprovedModelConfig
from lrd.standard.config import ModelCofig

TRAIN, TEST = 600, 48


@pytest.fixture(scope="module")
def data():
    return load_fama_french(cmp.DATA_PATH)


@pytest.fixture(scope="module")
def runs(data):
    X, y, dates = data
    experiment = cmp.improved_experiment([
        ImprovedModelConfig("ridge", "ridge", ["standardize"], grid={"alpha": [0.1, 10.0]}),
    ])
    specs = [
        cmp.standard_spec(ModelCofig(model_type="global_linear")),
        cmp.standard_spec(ModelCofig(model_type="mc_linear", n_simulations=10, subsample_size=60, random_state=0)),
        cmp.improved_spec(experiment.models[0], experiment),  # historical-mean benchmark
        cmp.improved_spec(experiment.models[1], experiment),
    ]
    return {s.key: cmp.run_model(s, X, y, dates, TRAIN, TEST) for s in specs}


def test_default_specs_cover_both_systems():
    keys = [s.key for s in cmp.default_specs()]
    assert len(keys) == 10 and len(set(keys)) == 10
    assert cmp.BENCHMARK_KEY in keys
    assert sum(k.startswith("standard:") for k in keys) == 7


def test_runs_share_months_and_shapes(runs, data):
    _, y, _ = data
    n_folds = (len(y) - TRAIN) // TEST
    for run in runs.values():
        assert len(run.pred) == len(run.y_true) == n_folds * TEST
        assert len(run.folds) == n_folds
        assert run.protocol == "walk_forward"
    dates = [r.dates for r in runs.values()]
    assert all((d == dates[0]).all() for d in dates)


def test_uncertainty_and_tuning_recorded(runs):
    assert runs["standard:global_linear"].std is None
    assert runs["standard:mc_linear"].has_uncertainty
    tuned = runs["improved:ridge"]
    assert tuned.params["grid"] == {"alpha": [0.1, 10.0]}
    assert len(tuned.tuning) == len(tuned.folds)
    assert set(tuned.tuning["chosen"].iloc[0]) == {"alpha"}


def test_summarize(runs):
    bench = runs[cmp.BENCHMARK_KEY]
    summary = cmp.summarize(list(runs.values()), bench).set_index("key")
    assert summary.loc[cmp.BENCHMARK_KEY, "oos_r2_%"] == 0.0
    assert summary["folds_won"].sum() == len(bench.folds)
    cov = summary.loc["improved:ridge", ["coverage_50", "coverage_80", "coverage_90", "coverage_95"]].to_numpy()
    assert (np.diff(cov) >= 0).all()
    assert np.isnan(summary.loc["standard:global_linear", "coverage_95"])
    assert summary["rmse"].notna().all()


def test_misaligned_runs_rejected(runs, data):
    X, y, dates = data
    short = cmp.run_model(cmp.standard_spec(ModelCofig(model_type="global_linear")), X, y, dates, 500, 48)
    with pytest.raises(ValueError, match="same months"):
        cmp.summarize([short], runs[cmp.BENCHMARK_KEY])


def test_pairwise_dm_is_antisymmetric(runs):
    t, p = cmp.pairwise_dm(list(runs.values()))
    np.testing.assert_allclose(t.to_numpy(), -t.to_numpy().T, atol=1e-12)
    np.testing.assert_allclose(p.to_numpy(), p.to_numpy().T, atol=1e-12)
    assert (np.diag(p.to_numpy()) == 1.0).all()


def test_cumulative_advantage_matches_sse(runs):
    run, bench = runs["improved:ridge"], runs[cmp.BENCHMARK_KEY]
    adv = cmp.cumulative_advantage(run, bench)
    assert adv.iloc[-1] == pytest.approx(np.sum(bench.errors**2) - np.sum(run.errors**2))


def test_holdout(data):
    X, y, dates = data
    run = cmp.run_holdout(cmp.standard_spec(ModelCofig(model_type="ridge_linear", alpha=1.0)), X, y, dates)
    assert run.protocol == "holdout"
    assert len(run.pred) == len(y) - int(len(y) * 0.8)
    assert len(run.folds) == 1


def test_save_load_roundtrip(runs, tmp_path):
    cmp.save_runs(list(runs.values()), tmp_path / "run", {"note": "test"})
    loaded, meta = cmp.load_runs(tmp_path / "run")
    assert meta["note"] == "test" and meta["data_hash"] == cmp.data_hash()
    by_key = {r.key: r for r in loaded}
    for key, original in runs.items():
        restored = by_key[key]
        np.testing.assert_allclose(restored.pred, original.pred)
        assert (restored.std is None) == (original.std is None)
        assert restored.params == original.params
        pd.testing.assert_series_equal(restored.folds["rmse"], original.folds["rmse"], check_names=False)
    assert by_key["improved:ridge"].tuning["chosen"].iloc[0] == runs["improved:ridge"].tuning["chosen"].iloc[0]


def test_shipped_dashboard_results_are_current():
    """app/data/default_run must be rebuilt (scripts/build_app_results.py) whenever the data changes."""
    loaded, meta = cmp.load_runs("app/data/default_run")
    assert meta["data_hash"] == cmp.data_hash()
    assert {r.key for r in loaded} == {s.key for s in cmp.default_specs()}
