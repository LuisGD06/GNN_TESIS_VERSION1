import streamlit as st
import plotly.express as px

from app.utils.data_loader import load_subgraph_alerts


st.set_page_config(page_title="Análisis de subgrafos", layout="wide")

st.title("Análisis estructural de alertas")

subgraph_alerts_df = load_subgraph_alerts()

if subgraph_alerts_df is None:
    st.error("No se encontró subgraph_alerts.parquet. Ejecuta el Nivel 6.")
    st.stop()

st.sidebar.header("Filtros")

label_options = sorted(subgraph_alerts_df["label"].dropna().unique())
selected_labels = st.sidebar.multiselect(
    "Etiqueta",
    options=label_options,
    default=label_options,
)

priority_options = sorted(subgraph_alerts_df["priority_level"].dropna().unique())
selected_priority = st.sidebar.multiselect(
    "Prioridad",
    options=priority_options,
    default=priority_options,
)

filtered_df = subgraph_alerts_df[
    subgraph_alerts_df["label"].isin(selected_labels)
    & subgraph_alerts_df["priority_level"].isin(selected_priority)
].copy()

st.subheader("Alertas con features de subgrafos")

st.dataframe(
    filtered_df,
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

st.subheader("Ratio unknown en vecindarios")

ratio_cols = [
    col for col in filtered_df.columns
    if "ratio_unknown_neighbors" in col
]

if ratio_cols:
    melted_unknown = filtered_df[
        ["txId", "label", "priority_level"] + ratio_cols
    ].melt(
        id_vars=["txId", "label", "priority_level"],
        var_name="feature",
        value_name="ratio_unknown"
    )

    fig_unknown = px.box(
        melted_unknown,
        x="feature",
        y="ratio_unknown",
        color="label",
        title="Distribución de ratio unknown neighbors en alertas",
    )

    st.plotly_chart(fig_unknown, use_container_width=True)

st.info(
    "Estas variables estructurales no reemplazan al score del modelo. Sirven para interpretar el contexto local de las alertas."
)