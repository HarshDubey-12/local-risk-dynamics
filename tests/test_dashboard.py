"""Smoke and interaction tests for the dashboard (app/) using Streamlit's AppTest."""

import pytest
from streamlit.testing.v1 import AppTest

from app.catalog import MODELS
from lrd import comparison as cmp

TIMEOUT = 300


def _screen(name: str) -> None:
    from app.views import comparison, data_analysis, models, overview

    {"overview": overview, "data": data_analysis, "models": models, "comparison": comparison}[name].render()


def _run(name: str, session: dict | None = None) -> AppTest:
    at = AppTest.from_function(_screen, args=(name,), default_timeout=TIMEOUT)
    for key, value in (session or {}).items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_entry_point_defines_four_screens():
    from app import dashboard

    assert callable(dashboard.run)


@pytest.mark.parametrize("screen", ["overview", "data", "models", "comparison"])
def test_screens_render(screen):
    at = _run(screen)
    assert at.title


def test_catalog_covers_every_default_model():
    assert set(MODELS) == {s.key for s in cmp.default_specs()}


def test_overview_headline_and_cards():
    at = _run("overview")
    assert any(m.value.startswith("Best model: **MCLLR (improved)**") for m in at.markdown)
    assert "OOS R² vs historical mean" in [m.label for m in at.metric]
    assert len([b for b in at.button if b.label == "Open model →"]) == len(MODELS)


def test_data_screen_interactions():
    at = _run("data")
    assert any("PASSED" in m.value for m in at.markdown)
    at.slider[1].set_value(3.0).run()  # outlier threshold
    assert not at.exception
    at.multiselect[0].set_value([]).run()
    assert any("Select at least one factor" in w.value for w in at.warning)


def test_every_model_page_renders():
    at = _run("models")
    for key in MODELS:
        at.selectbox[0].set_value(key).run()
        assert not at.exception, (key, [e.value for e in at.exception])
        assert at.header[0].value == MODELS[key]["label"]


def test_custom_standard_run_walk_forward_and_holdout():
    at = _run("models", {"model_key": "standard:ridge_linear"})
    at.radio[0].set_value("Custom settings").run()
    at.select_slider[0].set_value(0.1)
    at.button[0].click().run()
    assert not at.exception
    assert any("added to the Comparison" in s.value for s in at.success)
    assert len(at.session_state["custom_runs"]) == 1

    at.radio[1].set_value("Hold-out 80/20 (secondary view)")
    at.button[0].click().run()
    assert not at.exception
    assert any("Hold-out evaluation" in m.value for m in at.markdown)
    assert len(at.session_state["custom_runs"]) == 1  # hold-out runs are not added to the comparison


def test_custom_improved_fixed_run():
    at = _run("models", {"model_key": "improved:mcllr_improved"})
    at.radio[0].set_value("Custom settings").run()
    at.radio[1].set_value("Fixed values").run()
    at.slider[0].set_value(10)  # n_simulations
    at.button[0].click().run()
    assert not at.exception
    key = next(iter(at.session_state["custom_runs"]))
    assert key.startswith("improved:mcllr_improved#")


def test_comparison_includes_custom_runs():
    from lrd.core.data.loader import load_fama_french
    from lrd.standard.config import ModelConfig

    X, y, dates = load_fama_french(cmp.DATA_PATH)
    run = cmp.run_model(cmp.standard_spec(ModelConfig(model_type="ridge_linear", alpha=0.1), key="standard:ridge_linear#test"),
                        X, y, dates)
    at = _run("comparison", {"custom_runs": {run.key: (run, "Ridge custom α=0.1")}})
    assert "standard:ridge_linear#test" in at.multiselect[0].value


def test_comparison_needs_two_models():
    at = _run("comparison")
    at.multiselect[0].set_value([cmp.BENCHMARK_KEY]).run()
    assert any("at least two" in w.value for w in at.warning)
