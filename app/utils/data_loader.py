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

ERROR_CONFUSION_SUMMARY_PATH = REPORTS_DIR / "tables" / "error_confusion_summary.csv"
ERROR_FALSE_POSITIVES_PATH = REPORTS_DIR / "tables" / "error_cases_false_positives_top.csv"
ERROR_FALSE_NEGATIVES_PATH = REPORTS_DIR / "tables" / "error_cases_false_negatives_top.csv"
ERROR_SCORE_BY_LABEL_PATH = REPORTS_DIR / "tables" / "error_score_by_label.csv"
ERROR_SCORE_BY_TIMESTEP_PATH = REPORTS_DIR / "tables" / "error_score_by_timestep.csv"
ERROR_PRIORITY_BY_LABEL_PATH = REPORTS_DIR / "tables" / "error_priority_by_label.csv"
ERROR_RECALL_AT_K_PATH = REPORTS_DIR / "tables" / "error_recall_at_k.csv"
ERROR_ANALYSIS_SUMMARY_PATH = REPORTS_DIR / "metrics" / "error_analysis_summary.json"


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

def load_error_confusion_summary() -> pd.DataFrame | None:
    if not file_exists(ERROR_CONFUSION_SUMMARY_PATH):
        return None
    return load_csv(str(ERROR_CONFUSION_SUMMARY_PATH))


def load_error_false_positives() -> pd.DataFrame | None:
    if not file_exists(ERROR_FALSE_POSITIVES_PATH):
        return None
    return load_csv(str(ERROR_FALSE_POSITIVES_PATH))


def load_error_false_negatives() -> pd.DataFrame | None:
    if not file_exists(ERROR_FALSE_NEGATIVES_PATH):
        return None
    return load_csv(str(ERROR_FALSE_NEGATIVES_PATH))


def load_error_score_by_label() -> pd.DataFrame | None:
    if not file_exists(ERROR_SCORE_BY_LABEL_PATH):
        return None
    return load_csv(str(ERROR_SCORE_BY_LABEL_PATH))


def load_error_score_by_timestep() -> pd.DataFrame | None:
    if not file_exists(ERROR_SCORE_BY_TIMESTEP_PATH):
        return None
    return load_csv(str(ERROR_SCORE_BY_TIMESTEP_PATH))


def load_error_priority_by_label() -> pd.DataFrame | None:
    if not file_exists(ERROR_PRIORITY_BY_LABEL_PATH):
        return None
    return load_csv(str(ERROR_PRIORITY_BY_LABEL_PATH))


def load_error_recall_at_k() -> pd.DataFrame | None:
    if not file_exists(ERROR_RECALL_AT_K_PATH):
        return None
    return load_csv(str(ERROR_RECALL_AT_K_PATH))


def load_error_analysis_summary() -> dict | None:
    if not file_exists(ERROR_ANALYSIS_SUMMARY_PATH):
        return None
    return load_json(str(ERROR_ANALYSIS_SUMMARY_PATH))