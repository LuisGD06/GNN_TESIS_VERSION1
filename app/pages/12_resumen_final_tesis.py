import sys
from pathlib import Path
import json

import pandas as pd
import streamlit as st
import plotly.express as px


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SUMMARY_PATH = PROJECT_ROOT / "reports" / "metrics" / "final_thesis_analysis_summary.json"
KEY_RESULTS_PATH = PROJECT_ROOT / "reports" / "tables" / "final_thesis_key_results.csv"


st.set_page_config(page_title="Resumen final tesis", layout="wide")


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


st.title("Resumen final del análisis de tesis")

st.caption(
    "Vista consolidada de resultados técnicos, métricas, validación, explicabilidad y estado del prototipo AML."
)

if not SUMMARY_PATH.exists() or not KEY_RESULTS_PATH.exists():
    st.error(
        "Faltan los artefactos de resumen final. "
        "Ejecuta primero: python scripts/14_generate_final_thesis_summary.py"
    )
    st.stop()


summary = load_json(SUMMARY_PATH)
key_results_df = load_csv(KEY_RESULTS_PATH)

primary_model = summary.get("primary_model", {})
platform_counts = summary.get("platform_counts", {})
temporal_test = summary.get("temporal_test_metrics", {})
technical_integrity = summary.get("technical_integrity", {})
explainability = summary.get("explainability", {})

st.info(summary.get("main_interpretation", ""))

st.subheader("Estado general del prototipo")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Estado", summary.get("project_status", "No disponible"))
col2.metric("Integridad técnica", technical_integrity.get("overall_status", "No disponible"))
col3.metric("Modelo principal", primary_model.get("name", "No disponible"))
col4.metric("Total predicciones", f"{platform_counts.get('total_predictions', 0):,}")

st.divider()

st.subheader("Métricas del modelo principal")

col_a, col_b, col_c, col_d, col_e = st.columns(5)

col_a.metric("PR-AUC test", f"{primary_model.get('test_pr_auc', 0):.4f}")
col_b.metric("ROC-AUC test", f"{primary_model.get('test_roc_auc', 0):.4f}")
col_c.metric("Precision test", f"{primary_model.get('test_precision', 0):.4f}")
col_d.metric("Recall test", f"{primary_model.get('test_recall', 0):.4f}")
col_e.metric("F1 test", f"{primary_model.get('test_f1', 0):.4f}")

st.markdown(
    """
El modelo principal se seleccionó usando principalmente PR-AUC, debido al desbalance entre transacciones lícitas e ilícitas.
"""
)

st.divider()

st.subheader("Ranking de modelos")

ranking_df = pd.DataFrame(summary.get("model_ranking_top_5", []))

if not ranking_df.empty:
    st.dataframe(ranking_df, width="stretch")

    fig_ranking = px.bar(
        ranking_df.sort_values("pr_auc", ascending=True),
        x="pr_auc",
        y="model",
        color="family",
        orientation="h",
        title="Top 5 modelos por PR-AUC",
    )

    st.plotly_chart(fig_ranking, width="stretch")
else:
    st.warning("No se encontró ranking de modelos.")

st.divider()

st.subheader("Predicciones y priorización de riesgo")

label_counts = platform_counts.get("label_counts", {})
priority_counts = platform_counts.get("priority_counts", {})

col_l1, col_l2, col_l3 = st.columns(3)

col_l1.metric("Illicit", f"{label_counts.get('illicit', 0):,}")
col_l2.metric("Licit", f"{label_counts.get('licit', 0):,}")
col_l3.metric("Unknown", f"{label_counts.get('unknown', 0):,}")

priority_df = pd.DataFrame(
    [
        {"priority_level": key, "count": value}
        for key, value in priority_counts.items()
    ]
)

if not priority_df.empty:
    fig_priority = px.bar(
        priority_df,
        x="priority_level",
        y="count",
        text="count",
        title="Distribución de prioridad de riesgo",
    )
    st.plotly_chart(fig_priority, width="stretch")

st.divider()

st.subheader("Validación temporal test")

col_t1, col_t2, col_t3, col_t4 = st.columns(4)

col_t1.metric("Total test", f"{temporal_test.get('total', 0):,}")
col_t2.metric("Precision", f"{temporal_test.get('precision', 0):.4f}")
col_t3.metric("Recall", f"{temporal_test.get('recall', 0):.4f}")
col_t4.metric("F1", f"{temporal_test.get('f1', 0):.4f}")

recall_at_k = summary.get("test_recall_at_k", {})

recall_df = pd.DataFrame(
    [
        {"k": int(float(k)), "recall_at_k": value}
        for k, value in recall_at_k.items()
    ]
).sort_values("k")

if not recall_df.empty:
    fig_recall = px.line(
        recall_df,
        x="k",
        y="recall_at_k",
        markers=True,
        title="Recall@K en split test",
    )

    st.plotly_chart(fig_recall, width="stretch")

st.divider()

st.subheader("Explicabilidad XGBoost")

top_shap = explainability.get("top_10_shap_features", [])
top_model = explainability.get("top_10_model_importance_features", [])

col_x1, col_x2 = st.columns(2)

with col_x1:
    st.markdown("### Top 10 SHAP")
    st.code("\n".join(top_shap), language="text")

with col_x2:
    st.markdown("### Top 10 importancia interna")
    st.code("\n".join(top_model), language="text")

st.markdown(
    """
Las variables más influyentes pertenecen principalmente al conjunto original anonimizado del dataset Elliptic.
Esto sugiere que el modelo principal obtiene gran parte de su capacidad predictiva de las features transaccionales,
mientras que los grafos y subgrafos complementan la interpretación estructural del riesgo.
"""
)

st.divider()

st.subheader("Tabla consolidada de resultados")

section_options = ["Todas"] + sorted(key_results_df["section"].dropna().unique().tolist())

selected_section = st.selectbox(
    "Filtrar por sección",
    options=section_options,
)

filtered_results_df = key_results_df.copy()

if selected_section != "Todas":
    filtered_results_df = filtered_results_df[
        filtered_results_df["section"] == selected_section
    ]

st.dataframe(filtered_results_df, width="stretch")

st.divider()

st.subheader("Limitaciones principales")

for limitation in summary.get("main_limitations", []):
    st.markdown(f"- {limitation}")

st.divider()
