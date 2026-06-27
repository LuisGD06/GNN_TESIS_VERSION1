import sys
from pathlib import Path
import math

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "elliptic" / "platform_predictions.parquet"
EDGES_PATH = PROJECT_ROOT / "data" / "processed" / "elliptic" / "platform_edges.parquet"


st.set_page_config(page_title="Vecindario transaccional", layout="wide")


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(PREDICTIONS_PATH)
    df["txId_str"] = df["txId"].astype(str).str.strip()
    return df


@st.cache_data(show_spinner=False)
def load_edges() -> pd.DataFrame:
    df = pd.read_parquet(EDGES_PATH)
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    return df


def risk_color(label: str, priority: str) -> str:
    label = str(label).lower()
    priority = str(priority).lower()

    if label == "illicit":
        return "#e74c3c"
    if label == "licit":
        return "#2ecc71"
    if priority in ["critical", "high"]:
        return "#f39c12"
    return "#95a5a6"


def build_local_graph(edges_df: pd.DataFrame, center_txid: str, max_neighbors: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    incoming = edges_df[edges_df["target"] == center_txid].copy()
    outgoing = edges_df[edges_df["source"] == center_txid].copy()

    incoming["direction"] = "incoming"
    outgoing["direction"] = "outgoing"

    local_edges = pd.concat([incoming, outgoing], ignore_index=True)

    if len(local_edges) > max_neighbors:
        local_edges = local_edges.head(max_neighbors).copy()

    node_ids = set([center_txid])
    node_ids.update(local_edges["source"].tolist())
    node_ids.update(local_edges["target"].tolist())

    nodes_df = pd.DataFrame({"txId_str": sorted(node_ids)})

    return nodes_df, local_edges


def add_node_positions(nodes_df: pd.DataFrame, local_edges: pd.DataFrame, center_txid: str) -> pd.DataFrame:
    nodes_df = nodes_df.copy()

    incoming_neighbors = (
        local_edges[local_edges["target"] == center_txid]["source"]
        .drop_duplicates()
        .tolist()
    )

    outgoing_neighbors = (
        local_edges[local_edges["source"] == center_txid]["target"]
        .drop_duplicates()
        .tolist()
    )

    positions = {
        center_txid: (0.0, 0.0)
    }

    # Entrantes a la izquierda.
    for idx, node in enumerate(incoming_neighbors):
        angle = 2 * math.pi * idx / max(len(incoming_neighbors), 1)
        positions[node] = (-1.5, math.sin(angle))

    # Salientes a la derecha.
    for idx, node in enumerate(outgoing_neighbors):
        angle = 2 * math.pi * idx / max(len(outgoing_neighbors), 1)
        positions[node] = (1.5, math.sin(angle))

    # En caso de nodos aislados o repetidos no posicionados.
    for idx, node in enumerate(nodes_df["txId_str"]):
        if node not in positions:
            positions[node] = (0.0, -1.5 - idx * 0.1)

    nodes_df["x"] = nodes_df["txId_str"].map(lambda node: positions[node][0])
    nodes_df["y"] = nodes_df["txId_str"].map(lambda node: positions[node][1])

    return nodes_df


def build_plot(nodes_df: pd.DataFrame, local_edges: pd.DataFrame, center_txid: str) -> go.Figure:
    position_map = {
        row["txId_str"]: (row["x"], row["y"])
        for _, row in nodes_df.iterrows()
    }

    edge_x = []
    edge_y = []

    for _, edge in local_edges.iterrows():
        x0, y0 = position_map[edge["source"]]
        x1, y1 = position_map[edge["target"]]

        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1),
        hoverinfo="none",
        mode="lines",
        name="aristas",
    )

    node_x = nodes_df["x"].tolist()
    node_y = nodes_df["y"].tolist()

    node_text = []

    for _, row in nodes_df.iterrows():
        node_text.append(
            f"txId: {row.get('txId_str', '')}<br>"
            f"label: {row.get('label', 'sin dato')}<br>"
            f"score: {row.get('score_illicit', 0):.4f}<br>"
            f"priority: {row.get('priority_level', 'sin dato')}<br>"
            f"total_degree: {row.get('total_degree', 'sin dato')}"
        )

    node_colors = [
        "#3498db" if row["txId_str"] == center_txid
        else risk_color(row.get("label", ""), row.get("priority_level", ""))
        for _, row in nodes_df.iterrows()
    ]

    node_sizes = [
        28 if row["txId_str"] == center_txid else 16
        for _, row in nodes_df.iterrows()
    ]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[
            "central" if row["txId_str"] == center_txid else ""
            for _, row in nodes_df.iterrows()
        ],
        textposition="top center",
        hoverinfo="text",
        hovertext=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1),
        ),
        name="nodos",
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title="Vecindario transaccional local",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=650,
    )

    return fig


st.title("Visualización del vecindario transaccional")

st.caption(
    "Explora el vecindario local de una transacción: entradas, salidas, score de riesgo, prioridad y etiqueta conocida."
)

if not PREDICTIONS_PATH.exists() or not EDGES_PATH.exists():
    st.error(
        "Faltan artefactos para visualizar el grafo. "
        "Ejecuta primero: python scripts/10_generate_graph_visualization_artifacts.py"
    )
    st.stop()


predictions_df = load_predictions()
edges_df = load_edges()

st.info(
    """
Esta visualización muestra un vecindario local de primer nivel. 
El objetivo es apoyar la interpretación del score mediante contexto relacional, no reconstruir toda la blockchain.
"""
)

col_input_1, col_input_2 = st.columns([2, 1])

with col_input_1:
    default_txid = str(predictions_df.sort_values("risk_rank").iloc[0]["txId"])

    txid_input = st.text_input(
        "Ingrese un txId",
        value=default_txid,
    )

with col_input_2:
    max_neighbors = st.slider(
        "Máximo de aristas vecinas",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
    )


center_txid = str(txid_input).strip()

if center_txid == "":
    st.warning("Ingrese un txId válido.")
    st.stop()


center_row = predictions_df[predictions_df["txId_str"] == center_txid]

if center_row.empty:
    st.error("El txId ingresado no existe dentro de platform_predictions.parquet.")
    st.stop()


center_info = center_row.iloc[0]

incoming_count = int((edges_df["target"] == center_txid).sum())
outgoing_count = int((edges_df["source"] == center_txid).sum())

st.subheader("Resumen de la transacción central")

col1, col2, col3, col4 = st.columns(4)

col1.metric("txId", center_txid)
col2.metric("Etiqueta", str(center_info.get("label", "sin dato")))
col3.metric("Score ilícito", f"{float(center_info.get('score_illicit', 0)):.4f}")
col4.metric("Prioridad", str(center_info.get("priority_level", "sin dato")))

col5, col6, col7, col8 = st.columns(4)

col5.metric("Risk rank", f"{int(center_info.get('risk_rank', 0)):,}")
col6.metric("In-degree", f"{int(center_info.get('in_degree', 0)):,}")
col7.metric("Out-degree", f"{int(center_info.get('out_degree', 0)):,}")
col8.metric("Vecinos directos", f"{incoming_count + outgoing_count:,}")

nodes_df, local_edges = build_local_graph(edges_df, center_txid, max_neighbors)

if local_edges.empty:
    st.warning("Esta transacción no tiene aristas directas registradas en el artefacto de plataforma.")
    st.stop()


nodes_df = nodes_df.merge(
    predictions_df[
        [
            "txId_str",
            "txId",
            "label",
            "timestep",
            "score_illicit",
            "priority_level",
            "risk_rank",
            "in_degree",
            "out_degree",
            "total_degree",
        ]
    ],
    on="txId_str",
    how="left",
)

nodes_df = add_node_positions(nodes_df, local_edges, center_txid)

st.subheader("Grafo local")

fig = build_plot(nodes_df, local_edges, center_txid)

st.plotly_chart(fig, width="stretch")

st.markdown(
    """
**Lectura del grafo:**  
El nodo azul representa la transacción central. Los nodos verdes son lícitos, los rojos ilícitos, los naranjas son unknown o de prioridad alta/crítica, y los grises representan nodos sin señales relevantes o sin etiqueta destacada.
"""
)

st.divider()

st.subheader("Aristas del vecindario")

edge_table = local_edges.copy()

edge_table = edge_table.merge(
    predictions_df[
        [
            "txId_str",
            "label",
            "score_illicit",
            "priority_level",
            "risk_rank",
        ]
    ].rename(
        columns={
            "txId_str": "source",
            "label": "source_label",
            "score_illicit": "source_score",
            "priority_level": "source_priority",
            "risk_rank": "source_risk_rank",
        }
    ),
    on="source",
    how="left",
)

edge_table = edge_table.merge(
    predictions_df[
        [
            "txId_str",
            "label",
            "score_illicit",
            "priority_level",
            "risk_rank",
        ]
    ].rename(
        columns={
            "txId_str": "target",
            "label": "target_label",
            "score_illicit": "target_score",
            "priority_level": "target_priority",
            "risk_rank": "target_risk_rank",
        }
    ),
    on="target",
    how="left",
)

st.dataframe(edge_table, width="stretch")

st.divider()

st.subheader("Nodos del vecindario")

node_table_cols = [
    "txId",
    "label",
    "timestep",
    "score_illicit",
    "priority_level",
    "risk_rank",
    "in_degree",
    "out_degree",
    "total_degree",
]

node_table_cols = [
    col for col in node_table_cols
    if col in nodes_df.columns
]

st.dataframe(
    nodes_df[node_table_cols].sort_values("score_illicit", ascending=False),
    width="stretch",
)

st.divider()

st.subheader("Análisis del vecindario")

num_nodes = len(nodes_df)
num_edges = len(local_edges)

num_illicit_neighbors = int((nodes_df["label"] == "illicit").sum())
num_licit_neighbors = int((nodes_df["label"] == "licit").sum())
num_unknown_neighbors = int((nodes_df["label"] == "unknown").sum())

mean_neighbor_score = float(nodes_df["score_illicit"].fillna(0).mean())
max_neighbor_score = float(nodes_df["score_illicit"].fillna(0).max())

col_a, col_b, col_c, col_d = st.columns(4)

col_a.metric("Nodos visibles", f"{num_nodes:,}")
col_b.metric("Aristas visibles", f"{num_edges:,}")
col_c.metric("Illicit visibles", f"{num_illicit_neighbors:,}")
col_d.metric("Unknown visibles", f"{num_unknown_neighbors:,}")

col_e, col_f = st.columns(2)

col_e.metric("Score promedio vecindario", f"{mean_neighbor_score:.4f}")
col_f.metric("Score máximo vecindario", f"{max_neighbor_score:.4f}")

if num_illicit_neighbors > 1:
    st.error(
        "El vecindario contiene nodos ilícitos conocidos. "
        "Se recomienda revisión prioritaria del contexto transaccional."
    )
elif num_unknown_neighbors > 0 and max_neighbor_score >= 0.80:
    st.warning(
        "El vecindario contiene nodos unknown con score alto. "
        "Estos casos deben revisarse como señales de riesgo no confirmadas."
    )
else:
    st.success(
        "El vecindario mostrado no concentra señales críticas adicionales según las etiquetas y scores disponibles."
    )


st.divider()

st.subheader("Texto interpretativo para tesis")

st.markdown(
    """
La visualización del vecindario transaccional permite complementar el score predictivo con información relacional.
En lugar de analizar una transacción de forma aislada, el prototipo muestra sus conexiones directas, distinguiendo entradas,
salidas, etiquetas conocidas, prioridad y score de riesgo. Esta vista resulta relevante para un contexto AML, porque las
actividades ilícitas pueden manifestarse mediante patrones de flujo, concentración de relaciones, interacción con nodos
sin etiqueta confirmada o proximidad a transacciones previamente identificadas como ilícitas.
"""
)