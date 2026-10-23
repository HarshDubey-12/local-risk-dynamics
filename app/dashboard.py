"""
Local Risk Dynamics dashboard.

Run from the project root:
    streamlit run app/dashboard.py

Screens
-------
1. Project overview   objective, methodology, every model and the practices used
2. Data analysis      validation checks and interactive pre-analysis of the data
3. Models             each model (switchable): how it works, live settings, results
4. Comparison         any models from either system, compared on shared folds
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)  # dataset and config paths are project-relative
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from app.views import comparison, data_analysis, models, overview  # noqa: E402


def run() -> None:
    st.set_page_config(page_title="Local Risk Dynamics", page_icon="📈", layout="wide")
    pages = {
        "overview": st.Page(overview.render, title="Project overview", icon="🏠", url_path="overview", default=True),
        "data": st.Page(data_analysis.render, title="Data analysis", icon="🔎", url_path="data"),
        "models": st.Page(models.render, title="Models", icon="🧠", url_path="models"),
        "comparison": st.Page(comparison.render, title="Comparison", icon="⚖️", url_path="comparison"),
    }
    st.session_state["pages"] = pages
    st.sidebar.caption("Local Risk Dynamics · Fama–French 5-factor forecasting · standard vs improved systems")
    st.navigation(list(pages.values())).run()


# `streamlit run` executes this file as __main__; importing it only defines run().
if __name__ == "__main__":
    run()
