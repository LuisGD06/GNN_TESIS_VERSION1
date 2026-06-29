import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.data_loader import (
    load_platform_predictions,
    load_subgraph_alerts,
)


st.set_page_config(page_title="Detalle de transacción", layout="wide")

st.title("Detalle de transacción AML")
st.caption("Consulta individual de score, prioridad y contexto estructural de una transacción.")

predictions_df = load_platform_predictions()
subgraph_alerts_df = load_subgraph_alerts()

if predictions_df is None:
    st.error("No se encontró platform_predictions.parquet.")
    st.stop()

predictions_df = predictions_df.copy()
predictions_df["txId_str"] = predictions_df["txId"].astype(str)

st.sidebar.header("Búsqueda")

default_txid = str(
    predictions_df.sort_values("score_illicit", ascending=False)
    .iloc[0]["txId"]
)

txid_input = st.sidebar.text_input(
    "Ingrese txId",
    value=default_txid,
)

result = predictions_df[predictions_df["txId_str"] == txid_input.strip()]

if result.empty:
    st.warning("No se encontró la transacción ingresada.")
    st.stop()

row = result.iloc[0]

st.subheader("Resumen de riesgo")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("txId", str(row.get("txId", "N/A")))
col2.metric("Score ilícito", f"{row.get('score_illicit', 0):.4f}")
col3.metric("Prioridad", str(row.get("priority_level", "N/A")))
col4.metric("Etiqueta", str(row.get("label", "N/A")))
col5.metric("Ranking", str(row.get("risk_rank", "N/A")))

st.divider()

st.subheader("Características de grafo")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Timestep", str(row.get("timestep", "N/A")))
col2.metric("In-degree", str(row.get("in_degree", "N/A")))
col3.metric("Out-degree", str(row.get("out_degree", "N/A")))
col4.metric("Total-degree", str(row.get("total_degree", "N/A")))

st.subheader("Registro completo")

display_cols = [
    col for col in [
        "txId",
        "timestep",
        "label",
        "score_illicit",
        "predicted_class",
        "risk_level_probability",
        "priority_level",
        "risk_rank",
        "in_degree",
        "out_degree",
        "total_degree",
    ]
    if col in result.columns
]

st.dataframe(result[display_cols], width="stretch")

st.divider()

st.subheader("Contexto estructural de subgrafo")

if subgraph_alerts_df is not None:
    subgraph_alerts_df = subgraph_alerts_df.copy()
    subgraph_alerts_df["txId_str"] = subgraph_alerts_df["txId"].astype(str)

    subgraph_result = subgraph_alerts_df[
        subgraph_alerts_df["txId_str"] == txid_input.strip()
    ]

    if not subgraph_result.empty:
        structural_cols = [
            col for col in subgraph_result.columns
            if (
                "subgraph" in col
                or "ratio_unknown" in col
                or "ratio_illicit" in col
                or "neighbor" in col
                or col in ["txId", "label", "score_illicit", "priority_level"]
            )
        ]

        st.dataframe(
            subgraph_result[structural_cols],
            width="stretch",
        )
    else:
        st.info(
            "Esta transacción no se encuentra en el archivo de alertas top con subgrafos. "
            "El contexto estructural detallado solo está disponible para las alertas priorizadas."
        )
else:
    st.info("No se encontró subgraph_alerts.parquet.")

st.divider()

st.subheader("Interpretación")

score = float(row.get("score_illicit", 0))
label = str(row.get("label", "unknown"))
priority = str(row.get("priority_level", "N/A"))

if label == "unknown":
    st.warning(
        f"La transacción tiene etiqueta unknown y prioridad {priority}. "
        "Esto no significa que sea ilícita confirmada; indica que el modelo la prioriza para revisión."
    )
elif label == "illicit":
    st.error(
        f"La transacción está etiquetada como ilícita y obtuvo un score de {score:.4f}. "
        "Este caso puede utilizarse para analizar patrones detectados correctamente por el modelo."
    )
else:
    st.success(
        f"La transacción está etiquetada como lícita y obtuvo un score de {score:.4f}. "
        "Si el score es alto, puede representar un falso positivo o un caso estructuralmente riesgoso."
    )