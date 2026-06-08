from pathlib import Path
import json
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "elliptic"
REPORTS_DIR = PROJECT_ROOT / "reports"

PLATFORM_PREDICTIONS_PATH = PROCESSED_DIR / "platform_predictions.parquet"
PLATFORM_ALERTS_TOP_PATH = PROCESSED_DIR / "platform_alerts_top.parquet"
SUBGRAPH_ALERTS_PATH = PROCESSED_DIR / "subgraph_alerts.parquet"

FINAL_MODEL_COMPARISON_PATH = REPORTS_DIR / "tables" / "final_model_comparison.csv"
PLATFORM_METRICS_SUMMARY_PATH = REPORTS_DIR / "metrics" / "platform_metrics_summary.json"
PLATFORM_EXPORT_METADATA_PATH = REPORTS_DIR / "metrics" / "platform_export_metadata.json"


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_platform_predictions() -> pd.DataFrame | None:
    if not file_exists(PLATFORM_PREDICTIONS_PATH):
        return None
    return load_parquet(str(PLATFORM_PREDICTIONS_PATH))


def load_platform_alerts() -> pd.DataFrame | None:
    if not file_exists(PLATFORM_ALERTS_TOP_PATH):
        return None
    return load_parquet(str(PLATFORM_ALERTS_TOP_PATH))


def load_subgraph_alerts() -> pd.DataFrame | None:
    if not file_exists(SUBGRAPH_ALERTS_PATH):
        return None
    return load_parquet(str(SUBGRAPH_ALERTS_PATH))


def load_model_comparison() -> pd.DataFrame | None:
    if not file_exists(FINAL_MODEL_COMPARISON_PATH):
        return None
    return load_csv(str(FINAL_MODEL_COMPARISON_PATH))


def load_metrics_summary() -> dict | None:
    if not file_exists(PLATFORM_METRICS_SUMMARY_PATH):
        return None
    return load_json(str(PLATFORM_METRICS_SUMMARY_PATH))


def load_export_metadata() -> dict | None:
    if not file_exists(PLATFORM_EXPORT_METADATA_PATH):
        return None
    return load_json(str(PLATFORM_EXPORT_METADATA_PATH))


def check_required_files() -> dict:
    return {
        "platform_predictions.parquet": file_exists(PLATFORM_PREDICTIONS_PATH),
        "platform_alerts_top.parquet": file_exists(PLATFORM_ALERTS_TOP_PATH),
        "subgraph_alerts.parquet": file_exists(SUBGRAPH_ALERTS_PATH),
        "final_model_comparison.csv": file_exists(FINAL_MODEL_COMPARISON_PATH),
        "platform_metrics_summary.json": file_exists(PLATFORM_METRICS_SUMMARY_PATH),
        "platform_export_metadata.json": file_exists(PLATFORM_EXPORT_METADATA_PATH),
    }