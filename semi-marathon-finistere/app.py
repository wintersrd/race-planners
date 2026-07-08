"""Race Prediction Modeler Streamlit entrypoint."""

import os
import sys
from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def ensure_streamlit_runtime() -> None:
    """Re-exec the script through Streamlit when launched directly."""
    if get_script_run_ctx(suppress_warning=True) is not None:
        return
    os.execv(
        sys.executable,
        [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)],
    )


def configure_page() -> None:
    """Configure page metadata and shared styling once Streamlit is active."""
    st.set_page_config(
        page_title="Race Prediction Modeler",
        page_icon="🏃‍♀️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
    <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #f0f2f6;
            padding: 10px 10px 0px 10px;
            border-radius: 10px 10px 0px 0px;
        }

        .stTabs [data-baseweb="tab"] {
            font-size: 18px;
            font-weight: 600;
            padding: 15px 25px;
            background-color: #e8eaed;
            border-radius: 8px 8px 0px 0px;
            border: 2px solid transparent;
            transition: all 0.2s ease;
            color: #555;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: #dce3ea;
            color: #2E86AB;
        }

        .stTabs [aria-selected="true"] {
            background-color: white !important;
            border: 2px solid #2E86AB !important;
            border-bottom: 2px solid white !important;
            color: #2E86AB !important;
        }

        .stTabs [data-baseweb="tab-panel"] {
            background-color: white;
            padding: 20px;
            border: 2px solid #2E86AB;
            border-top: none;
            border-radius: 0px 0px 10px 10px;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


def main() -> None:
    configure_page()
    from race_planners.streamlit_general import render_general_planner

    render_general_planner(Path(SCRIPT_DIR).parent)


if __name__ == "__main__":
    ensure_streamlit_runtime()
    main()
