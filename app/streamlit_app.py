import sys
from pathlib import Path

import streamlit as st
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.data_loader import (
    check_required_files,
    load_metrics_summary,
    load_model_comparison,
)


st.set_page_config(
    page_title="AML-GNN Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AML-GNN Dashboard")
st.caption("Prototipo analítico para detección y priorización de transacciones ilícitas en Bitcoin.")

st.markdown(
    """
Este dashboard consolida los resultados del proyecto de tesis basado en el dataset Elliptic.
El sistema utiliza un modelo principal **XGBoost baseline** para generar scores de riesgo,
y complementa la interpretación con métricas comparativas, análisis de grafos y features de subgrafos.

La plataforma no reentrena modelos. Consume artefactos generados previamente durante el flujo metodológico.
"""
)

st.divider()

file_status = check_required_files()

st.subheader("Estado de artefactos requeridos")

status_df = [
    {"archivo": name, "disponible": "Sí" if exists else "No"}
    for name, exists in file_status.items()
]

st.dataframe(status_df, use_container_width=True)

missing_files = [name for name, exists in file_status.items() if not exists]

if missing_files:
    st.warning(
        "Faltan algunos archivos exportados. Ejecuta primero el Nivel 6 antes de usar el dashboard completo."
    )
    st.stop()

metrics_summary = load_metrics_summary()
model_comparison_df = load_model_comparison()

st.subheader("Modelo principal seleccionado")

if metrics_summary is not None:
    primary = metrics_summary.get("primary_model", {})

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Modelo", primary.get("name", "N/A"))
    col2.metric("PR-AUC test", f"{primary.get('test_pr_auc', 0):.4f}")
    col3.metric("ROC-AUC test", f"{primary.get('test_roc_auc', 0):.4f}")
    col4.metric("Precision test", f"{primary.get('test_precision', 0):.4f}")
    col5.metric("Recall test", f"{primary.get('test_recall', 0):.4f}")

st.subheader("Resumen comparativo de modelos")

if model_comparison_df is not None:
    st.dataframe(
        model_comparison_df.sort_values("pr_auc", ascending=False),
        use_container_width=True,
    )

st.info(
    "Nota: los scores asignados a nodos `unknown` representan priorización de riesgo, no etiquetas confirmadas."
)
