import sys
from pathlib import Path

import streamlit as st
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.data_loader import load_platform_predictions, load_platform_alerts


st.set_page_config(page_title="Predicciones", layout="wide")

st.title("Predicciones y alertas AML")
st.caption("Consulta de scores de riesgo y alertas priorizadas generadas por el modelo principal.")

predictions_df = load_platform_predictions()
alerts_df = load_platform_alerts()

if predictions_df is None:
    st.error("No se encontró platform_predictions.parquet. Ejecuta primero el Nivel 6.")
    st.stop()

st.sidebar.header("Filtros")

data_option = st.sidebar.radio(
    "Vista",
    options=["Todas las predicciones", "Top alertas"],
)

df = alerts_df.copy() if data_option == "Top alertas" and alerts_df is not None else predictions_df.copy()

label_options = sorted(df["label"].dropna().unique())
priority_options = sorted(df["priority_level"].dropna().unique())

selected_labels = st.sidebar.multiselect(
    "Etiqueta",
    options=label_options,
    default=label_options,
)

selected_priority = st.sidebar.multiselect(
    "Prioridad",
    options=priority_options,
    default=priority_options,
)

min_score = float(df["score_illicit"].min())
max_score = float(df["score_illicit"].max())

score_range = st.sidebar.slider(
    "Rango de score ilícito",
    min_value=min_score,
    max_value=max_score,
    value=(min_score, max_score),
)

filtered_df = df[
    df["label"].isin(selected_labels)
    & df["priority_level"].isin(selected_priority)
    & df["score_illicit"].between(score_range[0], score_range[1])
].copy()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Registros filtrados", f"{len(filtered_df):,}")
col2.metric("Score promedio", f"{filtered_df['score_illicit'].mean():.4f}" if len(filtered_df) else "0")
col3.metric("Ilícitas conocidas", f"{(filtered_df['label'] == 'illicit').sum():,}")
col4.metric("Unknown", f"{(filtered_df['label'] == 'unknown').sum():,}")

st.divider()

st.subheader("Tabla de predicciones")

columns_to_show = [
    "risk_rank",
    "txId",
    "timestep",
    "label",
    "score_illicit",
    "predicted_class",
    "risk_level_probability",
    "priority_level",
    "in_degree",
    "out_degree",
    "total_degree",
]

columns_to_show = [col for col in columns_to_show if col in filtered_df.columns]

st.dataframe(
    filtered_df[columns_to_show],
    use_container_width=True,
)

st.subheader("Distribución de scores")

fig_score = px.histogram(
    filtered_df,
    x="score_illicit",
    color="label",
    nbins=50,
    title="Distribución de score ilícito",
)

st.plotly_chart(fig_score, use_container_width=True)

st.subheader("Resumen por prioridad y etiqueta")

summary_df = (
    filtered_df
    .groupby(["priority_level", "label"])
    .size()
    .reset_index(name="count")
)

fig_summary = px.bar(
    summary_df,
    x="priority_level",
    y="count",
    color="label",
    barmode="group",
    title="Registros por prioridad y etiqueta",
)

st.plotly_chart(fig_summary, use_container_width=True)

st.warning(
    "Los nodos unknown con score alto no son ilícitos confirmados. "
    "Deben interpretarse como casos priorizados para revisión."
)