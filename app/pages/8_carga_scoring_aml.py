import sys
from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st
import plotly.express as px


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "elliptic" / "platform_predictions.parquet"
SUBGRAPH_ALERTS_PATH = PROJECT_ROOT / "data" / "processed" / "elliptic" / "subgraph_alerts.parquet"


st.set_page_config(page_title="Carga y scoring AML", layout="wide")


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"No se encontró {PREDICTIONS_PATH}")

    df = pd.read_parquet(PREDICTIONS_PATH)
    df["txId_str"] = df["txId"].astype(str).str.strip()

    return df


@st.cache_data(show_spinner=False)
def load_subgraph_alerts() -> pd.DataFrame | None:
    if not SUBGRAPH_ALERTS_PATH.exists():
        return None

    df = pd.read_parquet(SUBGRAPH_ALERTS_PATH)
    df["txId_str"] = df["txId"].astype(str).str.strip()

    return df


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if filename.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    raise ValueError("Formato no soportado. Usa CSV o XLSX.")


def build_template_csv() -> bytes:
    template = pd.DataFrame(
        {
            "txId": [
                "283172687",
                "209522745",
                "372726129",
                "999999999",
            ]
        }
    )

    return template.to_csv(index=False).encode("utf-8")


def build_template_xlsx() -> bytes:
    template = pd.DataFrame(
        {
            "txId": [
                "283172687",
                "209522745",
                "372726129",
                "999999999",
            ]
        }
    )

    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        template.to_excel(writer, index=False, sheet_name="txids")

    return output.getvalue()


def classify_result(row) -> str:
    if row["match_status"] != "Encontrado":
        return "No se encontró la transacción en los scores generados."

    score = row.get("score_illicit")

    if pd.isna(score):
        return "Score no disponible."

    if score >= 0.80:
        return "Riesgo alto: revisar con prioridad por posible patrón asociado a actividad ilícita."
    if score >= 0.50:
        return "Riesgo medio: revisar si existen señales adicionales en grafo o subgrafo."
    if score >= 0.20:
        return "Riesgo bajo-moderado: monitoreo sugerido según política AML."
    return "Riesgo bajo: no se prioriza salvo que existan reglas externas o señales adicionales."


def prepare_download_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def prepare_download_xlsx(df: pd.DataFrame) -> bytes:
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="scoring_aml")

    return output.getvalue()


st.title("Carga de archivos y consulta/scoring AML")

st.caption(
    "Sube un archivo CSV o XLSX con una columna txId para consultar el score de riesgo generado por el modelo principal."
)

st.info(
    """
Este módulo corresponde a una consulta operativa sobre los scores ya generados por el modelo XGBoost.
No reemplaza la revisión humana ni constituye una decisión automática final. 
Los nodos unknown se interpretan como casos sin etiqueta confirmada, no como ilícitos reales.
"""
)

try:
    predictions_df = load_predictions()
    subgraph_alerts_df = load_subgraph_alerts()
except Exception as error:
    st.error(f"No se pudieron cargar los artefactos de plataforma: {error}")
    st.stop()


st.subheader("1. Descargar plantilla opcional")

col_template_1, col_template_2 = st.columns(2)

with col_template_1:
    st.download_button(
        label="Descargar plantilla CSV",
        data=build_template_csv(),
        file_name="plantilla_txids_aml.csv",
        mime="text/csv",
    )

with col_template_2:
    st.download_button(
        label="Descargar plantilla XLSX",
        data=build_template_xlsx(),
        file_name="plantilla_txids_aml.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


st.subheader("2. Subir archivo")

uploaded_file = st.file_uploader(
    "Carga un archivo con una columna llamada txId",
    type=["csv", "xlsx"],
)

if uploaded_file is None:
    st.warning("Carga un archivo CSV o XLSX para iniciar la consulta.")
    st.stop()


try:
    uploaded_df = read_uploaded_file(uploaded_file)
except Exception as error:
    st.error(f"No se pudo leer el archivo: {error}")
    st.stop()


st.markdown("### Vista previa del archivo cargado")
st.dataframe(uploaded_df.head(20), width="stretch")


if "txId" not in uploaded_df.columns:
    st.error(
        "El archivo debe contener una columna llamada exactamente txId. "
        "Revisa la plantilla descargable."
    )
    st.stop()


query_df = uploaded_df.copy()
query_df["txId_original"] = query_df["txId"]
query_df["txId_str"] = query_df["txId"].astype(str).str.strip()

query_df = query_df[query_df["txId_str"].notna()]
query_df = query_df[query_df["txId_str"] != ""]

if query_df.empty:
    st.error("La columna txId no contiene valores válidos.")
    st.stop()


# Mantiene el orden original y permite identificar duplicados.
query_df["input_order"] = range(1, len(query_df) + 1)


selected_prediction_cols = [
    "txId_str",
    "txId",
    "timestep",
    "label",
    "in_degree",
    "out_degree",
    "total_degree",
    "score_illicit",
    "predicted_class",
    "risk_level_probability",
    "risk_percentile",
    "priority_level",
    "risk_rank",
]

selected_prediction_cols = [
    col for col in selected_prediction_cols
    if col in predictions_df.columns
]


result_df = query_df.merge(
    predictions_df[selected_prediction_cols],
    on="txId_str",
    how="left",
    suffixes=("_input", "_platform"),
)


result_df["match_status"] = result_df["score_illicit"].apply(
    lambda value: "No encontrado" if pd.isna(value) else "Encontrado"
)

result_df["risk_comment"] = result_df.apply(classify_result, axis=1)


# Agrega algunas señales de subgrafo solo si la transacción está dentro de las top alertas con subgrafo.
if subgraph_alerts_df is not None:
    subgraph_cols = [
        "txId_str",
        "subgraph_num_nodes_k1",
        "subgraph_num_nodes_k2",
        "subgraph_density_k1",
        "subgraph_density_k2",
        "ratio_unknown_neighbors_k1",
        "ratio_unknown_neighbors_k2",
        "ratio_illicit_neighbors_k1",
        "ratio_illicit_neighbors_k2",
        "fan_in_out_ratio_k1",
        "fan_in_out_ratio_k2",
    ]

    subgraph_cols = [
        col for col in subgraph_cols
        if col in subgraph_alerts_df.columns
    ]

    result_df = result_df.merge(
        subgraph_alerts_df[subgraph_cols],
        on="txId_str",
        how="left",
    )


# Orden final de columnas para visualización.
preferred_cols = [
    "input_order",
    "txId_original",
    "match_status",
    "label",
    "timestep",
    "score_illicit",
    "risk_percentile",
    "priority_level",
    "risk_rank",
    "predicted_class",
    "in_degree",
    "out_degree",
    "total_degree",
    "risk_comment",
    "subgraph_num_nodes_k1",
    "subgraph_num_nodes_k2",
    "subgraph_density_k1",
    "subgraph_density_k2",
    "ratio_unknown_neighbors_k1",
    "ratio_unknown_neighbors_k2",
    "ratio_illicit_neighbors_k1",
    "ratio_illicit_neighbors_k2",
    "fan_in_out_ratio_k1",
    "fan_in_out_ratio_k2",
]

available_cols = [
    col for col in preferred_cols
    if col in result_df.columns
]

result_display_df = result_df[available_cols].copy()


st.subheader("3. Resumen de consulta")

total_uploaded = len(query_df)
total_found = int((result_df["match_status"] == "Encontrado").sum())
total_not_found = int((result_df["match_status"] == "No encontrado").sum())

critical_count = int((result_df["priority_level"] == "critical").sum()) if "priority_level" in result_df.columns else 0
high_count = int((result_df["priority_level"] == "high").sum()) if "priority_level" in result_df.columns else 0
medium_count = int((result_df["priority_level"] == "medium").sum()) if "priority_level" in result_df.columns else 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("Registros cargados", f"{total_uploaded:,}")
col2.metric("Encontrados", f"{total_found:,}")
col3.metric("No encontrados", f"{total_not_found:,}")
col4.metric("Critical + High", f"{critical_count + high_count:,}")


col5, col6, col7 = st.columns(3)

col5.metric("Critical", f"{critical_count:,}")
col6.metric("High", f"{high_count:,}")
col7.metric("Medium", f"{medium_count:,}")


st.subheader("4. Resultados de scoring")

show_only_found = st.checkbox("Mostrar solo transacciones encontradas", value=False)

filtered_df = result_display_df.copy()

if show_only_found:
    filtered_df = filtered_df[filtered_df["match_status"] == "Encontrado"]

if "priority_level" in filtered_df.columns:
    priority_options = sorted(
        filtered_df["priority_level"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_priorities = st.multiselect(
        "Filtrar por prioridad",
        options=priority_options,
        default=priority_options,
    )

    if selected_priorities:
        filtered_df = filtered_df[
            filtered_df["priority_level"].astype(str).isin(selected_priorities)
        ]


st.dataframe(filtered_df, width="stretch")


st.subheader("5. Visualizaciones")

found_df = result_df[result_df["match_status"] == "Encontrado"].copy()

if found_df.empty:
    st.warning("No hay transacciones encontradas para graficar.")
else:
    col_chart_1, col_chart_2 = st.columns(2)

    with col_chart_1:
        if "priority_level" in found_df.columns:
            priority_counts = (
                found_df["priority_level"]
                .fillna("sin_prioridad")
                .value_counts()
                .reset_index()
            )

            priority_counts.columns = ["priority_level", "count"]

            fig_priority = px.bar(
                priority_counts,
                x="priority_level",
                y="count",
                text="count",
                title="Cantidad por nivel de prioridad",
            )

            st.plotly_chart(fig_priority, width="stretch")

    with col_chart_2:
        if "label" in found_df.columns:
            label_counts = (
                found_df["label"]
                .fillna("sin_etiqueta")
                .value_counts()
                .reset_index()
            )

            label_counts.columns = ["label", "count"]

            fig_label = px.bar(
                label_counts,
                x="label",
                y="count",
                text="count",
                title="Distribución de etiquetas conocidas",
            )

            st.plotly_chart(fig_label, width="stretch")

    if "score_illicit" in found_df.columns:
        fig_score = px.histogram(
            found_df,
            x="score_illicit",
            nbins=30,
            title="Distribución de score ilícito en el archivo cargado",
        )

        st.plotly_chart(fig_score, width="stretch")


st.subheader("6. Descargar resultados")

col_download_1, col_download_2 = st.columns(2)

with col_download_1:
    st.download_button(
        label="Descargar resultados CSV",
        data=prepare_download_csv(result_display_df),
        file_name="resultados_scoring_aml.csv",
        mime="text/csv",
    )

with col_download_2:
    st.download_button(
        label="Descargar resultados XLSX",
        data=prepare_download_xlsx(result_display_df),
        file_name="resultados_scoring_aml.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


st.divider()

st.subheader("Interpretación para el usuario")

st.markdown(
    """
- **Encontrado:** la transacción existe dentro de los artefactos generados por el prototipo.
- **No encontrado:** la transacción no forma parte del dataset procesado o no tiene score disponible.
- **score_illicit:** probabilidad estimada por el modelo principal de que la transacción sea ilícita.
- **priority_level:** prioridad operativa según percentiles del score.
- **risk_rank:** posición de la transacción en el ranking global de riesgo.
- **unknown:** transacción sin etiqueta confirmada; no debe interpretarse automáticamente como ilícita.
"""
)

st.divider()

st.subheader("7. Reporte operativo del archivo cargado")

found_for_report_df = result_df[result_df["match_status"] == "Encontrado"].copy()

if found_for_report_df.empty:
    st.warning(
        "No se encontraron transacciones del archivo dentro de los scores generados. "
        "No es posible construir un reporte operativo de riesgo."
    )
else:
    top_risky_df = (
        found_for_report_df
        .sort_values("score_illicit", ascending=False)
        .head(10)
        .copy()
    )

    num_unknown = int((found_for_report_df["label"] == "unknown").sum())
    num_illicit = int((found_for_report_df["label"] == "illicit").sum())
    num_licit = int((found_for_report_df["label"] == "licit").sum())

    max_score = float(found_for_report_df["score_illicit"].max())
    mean_score = float(found_for_report_df["score_illicit"].mean())

    top_txid = str(top_risky_df.iloc[0]["txId_platform"]) if "txId_platform" in top_risky_df.columns else str(top_risky_df.iloc[0]["txId"])
    top_score = float(top_risky_df.iloc[0]["score_illicit"])
    top_priority = str(top_risky_df.iloc[0]["priority_level"])

    operative_summary = f"""
**Resumen operativo AML**

Se cargaron **{total_uploaded:,}** registros para consulta. 
De ellos, **{total_found:,}** fueron encontrados dentro de los artefactos del prototipo y **{total_not_found:,}** no tuvieron coincidencia.

Entre las transacciones encontradas, se identificaron:

- **{critical_count:,}** transacciones con prioridad crítica.
- **{high_count:,}** transacciones con prioridad alta.
- **{medium_count:,}** transacciones con prioridad media.
- **{num_illicit:,}** transacciones con etiqueta conocida ilícita.
- **{num_licit:,}** transacciones con etiqueta conocida lícita.
- **{num_unknown:,}** transacciones sin etiqueta confirmada.

El score promedio del archivo consultado fue **{mean_score:.4f}**, mientras que el score máximo observado fue **{max_score:.4f}**.
La transacción con mayor prioridad dentro del archivo fue **{top_txid}**, con score **{top_score:.4f}** y nivel de prioridad **{top_priority}**.

Este resultado debe interpretarse como una priorización de revisión AML. Las transacciones con etiqueta unknown no deben asumirse automáticamente como ilícitas, sino como casos que requieren análisis adicional.
"""

    st.markdown(operative_summary)

    st.markdown("### Top 10 transacciones más riesgosas del archivo")

    top_cols = [
        col for col in [
            "txId_platform",
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
        if col in top_risky_df.columns
    ]

    st.dataframe(top_risky_df[top_cols], width="stretch")

    st.download_button(
        label="Descargar resumen operativo TXT",
        data=operative_summary.encode("utf-8"),
        file_name="resumen_operativo_aml.txt",
        mime="text/plain",
    )

    if critical_count + high_count > 0:
        st.error(
            "El archivo cargado contiene transacciones con prioridad critical/high. "
            "Se recomienda revisión prioritaria."
        )
    elif medium_count > 0:
        st.warning(
            "El archivo cargado contiene transacciones de prioridad media. "
            "Se recomienda revisión complementaria según política AML."
        )
    else:
        st.success(
            "No se identificaron transacciones de prioridad crítica o alta en el archivo cargado."
        )