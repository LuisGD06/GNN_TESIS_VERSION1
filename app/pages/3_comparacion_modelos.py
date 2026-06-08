import streamlit as st
import plotly.express as px

from app.utils.data_loader import load_model_comparison


st.set_page_config(page_title="Comparación de modelos", layout="wide")

st.title("Comparación final de modelos")

comparison_df = load_model_comparison()

if comparison_df is None:
    st.error("No se encontró final_model_comparison.csv. Ejecuta el Nivel 6.")
    st.stop()

comparison_df = comparison_df.sort_values("pr_auc", ascending=False)

st.subheader("Tabla comparativa")

st.dataframe(comparison_df, use_container_width=True)

st.subheader("Ranking por PR-AUC")

fig_pr_auc = px.bar(
    comparison_df,
    x="model",
    y="pr_auc",
    color="family",
    title="Comparación de modelos por PR-AUC",
)

fig_pr_auc.update_layout(xaxis_tickangle=-60)

st.plotly_chart(fig_pr_auc, use_container_width=True)

st.subheader("Precision vs Recall")

fig_precision_recall = px.scatter(
    comparison_df,
    x="recall",
    y="precision",
    size="pr_auc",
    color="family",
    hover_name="model",
    title="Precision vs Recall por modelo",
)

st.plotly_chart(fig_precision_recall, use_container_width=True)

st.markdown(
    """
**Interpretación metodológica:**

- XGBoost baseline fue seleccionado como modelo principal por obtener el mayor PR-AUC.
- Random Forest mostró un perfil conservador con alta precisión.
- Las GNN alcanzaron recall alto, pero con baja precisión.
- Las features de subgrafos aportaron valor descriptivo, aunque no mejoraron de forma sustancial el desempeño predictivo.
"""
)