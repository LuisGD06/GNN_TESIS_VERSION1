import streamlit as st
import pandas as pd
import plotly.express as px

from app.utils.data_loader import (
    load_platform_predictions,
    load_metrics_summary,
)


st.set_page_config(page_title="Dashboard AML", layout="wide")

st.title("Dashboard ejecutivo AML")

predictions_df = load_platform_predictions()
metrics_summary = load_metrics_summary()

if predictions_df is None:
    st.error("No se encontró platform_predictions.parquet. Ejecuta el Nivel 6.")
    st.stop()

total_transactions = len(predictions_df)
num_illicit_known = int((predictions_df["label"] == "illicit").sum())
num_licit_known = int((predictions_df["label"] == "licit").sum())
num_unknown = int((predictions_df["label"] == "unknown").sum())

critical_alerts = int((predictions_df["priority_level"] == "critical").sum())
high_alerts = int((predictions_df["priority_level"] == "high").sum())

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Transacciones", f"{total_transactions:,}")
col2.metric("Ilícitas conocidas", f"{num_illicit_known:,}")
col3.metric("Lícitas conocidas", f"{num_licit_known:,}")
col4.metric("Unknown", f"{num_unknown:,}")
col5.metric("Alertas critical/high", f"{critical_alerts + high_alerts:,}")

st.divider()

st.subheader("Distribución de prioridad")

priority_counts = (
    predictions_df
    .groupby(["priority_level", "label"])
    .size()
    .reset_index(name="count")
)

fig_priority = px.bar(
    priority_counts,
    x="priority_level",
    y="count",
    color="label",
    barmode="group",
    title="Distribución de prioridad por etiqueta",
)

st.plotly_chart(fig_priority, width="stretch")

st.subheader("Distribución de scores de riesgo")

fig_score = px.histogram(
    predictions_df,
    x="score_illicit",
    nbins=60,
    color="label",
    title="Distribución de score ilícito por etiqueta",
)

st.plotly_chart(fig_score, width="stretch")

st.subheader("Top 20 transacciones con mayor score")

top_20 = predictions_df.sort_values("score_illicit", ascending=False).head(20)

st.dataframe(
    top_20[
        [
            "risk_rank",
            "txId",
            "timestep",
            "label",
            "score_illicit",
            "priority_level",
            "in_degree",
            "out_degree",
            "total_degree",
        ]
    ],
    width="stretch",
)

if metrics_summary:
    st.info(
        "Modelo principal: "
        f"{metrics_summary.get('primary_model', {}).get('name', 'N/A')} | "
        "Los nodos unknown se interpretan como priorización de riesgo, no como etiquetas confirmadas."
    )