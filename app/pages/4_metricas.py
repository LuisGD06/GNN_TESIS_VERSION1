import sys
from pathlib import Path

import streamlit as st
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.data_loader import load_model_comparison, load_metrics_summary


st.set_page_config(page_title="Métricas", layout="wide")

st.title("Métricas y comparación de modelos")
st.caption("Comparación final de modelos evaluados durante los niveles metodológicos.")

comparison_df = load_model_comparison()
metrics_summary = load_metrics_summary()

if comparison_df is None:
    st.error("No se encontró final_model_comparison.csv. Ejecuta primero el Nivel 6.")
    st.stop()

comparison_df = comparison_df.sort_values("pr_auc", ascending=False).reset_index(drop=True)

st.subheader("Modelo principal seleccionado")

if metrics_summary:
    primary = metrics_summary.get("primary_model", {})

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Modelo", primary.get("name", "N/A"))
    col2.metric("PR-AUC", f"{primary.get('test_pr_auc', 0):.4f}")
    col3.metric("ROC-AUC", f"{primary.get('test_roc_auc', 0):.4f}")
    col4.metric("Precision", f"{primary.get('test_precision', 0):.4f}")
    col5.metric("Recall", f"{primary.get('test_recall', 0):.4f}")

st.divider()

st.subheader("Tabla comparativa final")

st.dataframe(
    comparison_df,
    width="stretch",
)

st.subheader("Ranking por PR-AUC")

fig_pr_auc = px.bar(
    comparison_df,
    x="model",
    y="pr_auc",
    color="family",
    title="Comparación final de modelos por PR-AUC",
)

fig_pr_auc.update_layout(xaxis_tickangle=-70)

st.plotly_chart(fig_pr_auc, width="stretch")

st.subheader("Precision vs Recall")

fig_pr = px.scatter(
    comparison_df,
    x="recall",
    y="precision",
    size="pr_auc",
    color="family",
    hover_name="model",
    title="Relación precision-recall por modelo",
)

st.plotly_chart(fig_pr, width="stretch")

st.subheader("F1 vs PR-AUC")

fig_f1 = px.scatter(
    comparison_df,
    x="f1",
    y="pr_auc",
    size="roc_auc",
    color="family",
    hover_name="model",
    title="Relación F1 y PR-AUC",
)

st.plotly_chart(fig_f1, width="stretch")

st.markdown(
    """
### Interpretación

- **XGBoost baseline** fue seleccionado como modelo principal por obtener el mayor PR-AUC en test.
- **Random Forest** mostró un comportamiento más conservador, con alta precisión.
- Las **GNN** alcanzaron recall alto, pero presentaron baja precisión.
- Las features de **subgrafos** aportaron valor descriptivo, aunque no mejoraron sustancialmente el desempeño predictivo.
"""
)