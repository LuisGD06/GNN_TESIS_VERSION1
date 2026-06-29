import streamlit as st
import plotly.express as px

from app.utils.data_loader import load_platform_alerts


st.set_page_config(page_title="Alertas priorizadas", layout="wide")

st.title("Alertas AML priorizadas")

alerts_df = load_platform_alerts()

if alerts_df is None:
    st.error("No se encontró platform_alerts_top.parquet. Ejecuta el Nivel 6.")
    st.stop()

st.sidebar.header("Filtros")

priority_options = sorted(alerts_df["priority_level"].dropna().unique())
label_options = sorted(alerts_df["label"].dropna().unique())

selected_priority = st.sidebar.multiselect(
    "Prioridad",
    options=priority_options,
    default=priority_options,
)

selected_labels = st.sidebar.multiselect(
    "Etiqueta",
    options=label_options,
    default=label_options,
)

min_score = float(alerts_df["score_illicit"].min())
max_score = float(alerts_df["score_illicit"].max())

score_range = st.sidebar.slider(
    "Rango de score ilícito",
    min_value=min_score,
    max_value=max_score,
    value=(min_score, max_score),
)

filtered_df = alerts_df[
    alerts_df["priority_level"].isin(selected_priority)
    & alerts_df["label"].isin(selected_labels)
    & alerts_df["score_illicit"].between(score_range[0], score_range[1])
].copy()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Alertas filtradas", f"{len(filtered_df):,}")
col2.metric("Score promedio", f"{filtered_df['score_illicit'].mean():.4f}" if len(filtered_df) else "0")
col3.metric("Ilícitas conocidas", f"{(filtered_df['label'] == 'illicit').sum():,}")
col4.metric("Unknown", f"{(filtered_df['label'] == 'unknown').sum():,}")

st.divider()

st.subheader("Tabla de alertas")

st.dataframe(
    filtered_df[
        [
            "risk_rank",
            "txId",
            "timestep",
            "label",
            "score_illicit",
            "risk_level_probability",
            "priority_level",
            "in_degree",
            "out_degree",
            "total_degree",
        ]
    ],
    width="stretch",
)

st.subheader("Alertas por etiqueta y prioridad")

summary_df = (
    filtered_df
    .groupby(["label", "priority_level"])
    .size()
    .reset_index(name="count")
)

fig = px.bar(
    summary_df,
    x="priority_level",
    y="count",
    color="label",
    barmode="group",
    title="Alertas filtradas por prioridad y etiqueta",
)

st.plotly_chart(fig, width="stretch")

st.warning(
    "Las alertas con etiqueta unknown no son ilícitas confirmadas. Deben interpretarse como casos priorizados para revisión."
)