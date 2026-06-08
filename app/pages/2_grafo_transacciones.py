import sys
from pathlib import Path

import streamlit as st
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.data_loader import load_subgraph_alerts


st.set_page_config(page_title="Grafo de transacciones", layout="wide")

st.title("Grafo de transacciones y análisis estructural")
st.caption("Análisis de variables estructurales asociadas a las alertas priorizadas.")

subgraph_alerts_df = load_subgraph_alerts()

if subgraph_alerts_df is None:
    st.error("No se encontró subgraph_alerts.parquet. Ejecuta primero el Nivel 6.")
    st.stop()

st.sidebar.header("Filtros")

label_options = sorted(subgraph_alerts_df["label"].dropna().unique())
priority_options = sorted(subgraph_alerts_df["priority_level"].dropna().unique())

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

filtered_df = subgraph_alerts_df[
    subgraph_alerts_df["label"].isin(selected_labels)
    & subgraph_alerts_df["priority_level"].isin(selected_priority)
].copy()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Alertas analizadas", f"{len(filtered_df):,}")
col2.metric("Score promedio", f"{filtered_df['score_illicit'].mean():.4f}" if len(filtered_df) else "0")
col3.metric("Ilícitas conocidas", f"{(filtered_df['label'] == 'illicit').sum():,}")
col4.metric("Unknown", f"{(filtered_df['label'] == 'unknown').sum():,}")

st.divider()

st.subheader("Tabla de alertas con contexto estructural")

columns_to_show = [
    col for col in [
        "risk_rank",
        "txId",
        "timestep",
        "label",
        "score_illicit",
        "priority_level",
        "in_degree",
        "out_degree",
        "total_degree",
        "subgraph_num_nodes_k1",
        "subgraph_num_nodes_k2",
        "ratio_unknown_neighbors_k1",
        "ratio_unknown_neighbors_k2",
        "ratio_illicit_neighbors_k1",
        "ratio_illicit_neighbors_k2",
    ]
    if col in filtered_df.columns
]

st.dataframe(
    filtered_df[columns_to_show],
    use_container_width=True,
)

st.subheader("Tamaño de subgrafo k=1 vs k=2")

if "subgraph_num_nodes_k1" in filtered_df.columns and "subgraph_num_nodes_k2" in filtered_df.columns:
    fig_size = px.scatter(
        filtered_df,
        x="subgraph_num_nodes_k1",
        y="subgraph_num_nodes_k2",
        color="label",
        size="score_illicit",
        hover_name="txId",
        title="Tamaño de subgrafo k=1 vs k=2",
    )
    st.plotly_chart(fig_size, use_container_width=True)
else:
    st.info("No se encontraron columnas de tamaño de subgrafo k=1/k=2.")

st.subheader("Ratio de vecinos unknown")

ratio_unknown_cols = [
    col for col in filtered_df.columns
    if "ratio_unknown_neighbors" in col
]

if ratio_unknown_cols:
    melted_unknown = filtered_df[
        ["txId", "label", "priority_level"] + ratio_unknown_cols
    ].melt(
        id_vars=["txId", "label", "priority_level"],
        var_name="feature",
        value_name="ratio_unknown",
    )

    fig_unknown = px.box(
        melted_unknown,
        x="feature",
        y="ratio_unknown",
        color="label",
        title="Distribución de ratio unknown neighbors",
    )

    st.plotly_chart(fig_unknown, use_container_width=True)

st.subheader("Ratio de vecinos ilícitos")

ratio_illicit_cols = [
    col for col in filtered_df.columns
    if "ratio_illicit_neighbors" in col
]

if ratio_illicit_cols:
    melted_illicit = filtered_df[
        ["txId", "label", "priority_level"] + ratio_illicit_cols
    ].melt(
        id_vars=["txId", "label", "priority_level"],
        var_name="feature",
        value_name="ratio_illicit",
    )

    fig_illicit = px.box(
        melted_illicit,
        x="feature",
        y="ratio_illicit",
        color="label",
        title="Distribución de ratio illicit neighbors",
    )

    st.plotly_chart(fig_illicit, use_container_width=True)

st.info(
    "Estas variables estructurales ayudan a interpretar el contexto local de las alertas, "
    "pero no sustituyen al score predictivo del modelo principal."
)