#!/usr/bin/env python
"""Smoke tests for the Streamlit app (app/streamlit_app.py) using Streamlit's AppTest."""

import pytest
from streamlit.testing.v1 import AppTest

TIMEOUT = 120


def _page_script(page_name: str) -> None:
    from app import streamlit_app

    getattr(streamlit_app, page_name)()


def _run_page(page_name: str) -> AppTest:
    at = AppTest.from_function(_page_script, args=(page_name,), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("page", ["page_overview", "page_pipeline", "page_models"])
def test_explainer_pages_render(page):
    at = _run_page(page)
    assert at.title


def test_models_page_covers_every_model():
    from app import streamlit_app

    at = _run_page("page_models")
    for model_type in streamlit_app.MODEL_ORDER:
        at.selectbox[0].set_value(model_type).run()
        assert not at.exception
        assert at.subheader[0].value == streamlit_app.MODELS[model_type]["label"]


def test_locality_explorer_flags_global_bandwidth():
    at = _run_page("page_models")
    at.selectbox[0].set_value("lwlr").run()
    at.select_slider[1].set_value(1.0).run()
    assert not at.exception
    assert any("global counterpart" in w.value for w in at.warning)


def test_comparison_hold_out():
    at = _run_page("page_comparison")
    assert at.dataframe  # metrics table rendered with default settings
    assert not at.error


def test_comparison_walk_forward():
    at = _run_page("page_comparison")
    at.sidebar.multiselect[0].set_value(["global_linear", "mc_linear"])
    at.sidebar.radio[0].set_value("Walk-forward")
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert any("folds" in m.value for m in at.markdown)


def test_comparison_rejects_oversized_subsample():
    at = _run_page("page_comparison")
    at.sidebar.radio[0].set_value("Walk-forward")
    at.sidebar.slider[1].set_value(200)  # subsample m
    at.sidebar.slider[3].set_value(120)  # train window
    at.sidebar.button[0].click().run()
    assert any("exceeds" in e.value for e in at.error)
