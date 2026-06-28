import sys
from pathlib import Path
from io import BytesIO
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_PATH = PROJECT_ROOT / "models" / "baseline_xgboost.pkl"
SCHEMA_PATH = PROJECT_ROOT / "reports" / "metrics" / "inference_schema_xgboost.json"
TEMPLATE_PATH = PROJECT_ROOT / "data" / "processed" / "elliptic" / "inference_template_xgboost.csv"
REFERENCE_SCORES_PATH = PROJECT_ROOT / "data" / "processed" / "elliptic" / "platform_predictions.parquet"


st.set_page_config(page_title="Inferencia XGBoost", layout="wide")


@st.cache_data(show_spinner=False)
def load_schema() -> dict:
    if not SCHEMA_PATH.exists():
        return {
            "model_name": "XGBoost baseline",
            "required_features": [f"f_{i}" for i in range(165)]
            + ["in_degree", "out_degree", "total_degree"],
            "optional_columns": ["txId"],
        }

    with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def load_template() -> pd.DataFrame:
    if TEMPLATE_PATH.exists():
        return pd.read_csv(TEMPLATE_PATH)

    schema = load_schema()
    data = {"txId": ["ejemplo_1", "ejemplo_2"]}

    for col in schema["required_features"]:
        data[col] = [0.0, 0.0]

    return pd.DataFrame(data)


@st.cache_resource(show_spinner=False)
def load_model():
    try:
        import joblib
    except Exception as error:
        raise ImportError(
            "No se pudo importar joblib. Instala scikit-learn/joblib en el entorno."
        ) from error

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo {MODEL_PATH}. "
            "Este módulo requiere el modelo local baseline_xgboost.pkl."
        )

    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_reference_scores() -> np.ndarray | None:
    if not REFERENCE_SCORES_PATH.exists():
        return None

    df = pd.read_parquet(REFERENCE_SCORES_PATH)

    if "score_illicit" not in df.columns:
        return None

    return np.sort(df["score_illicit"].dropna().astype(float).values)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="inferencia_xgboost")

    return output.getvalue()


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if filename.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    raise ValueError("Formato no soportado. Usa CSV o XLSX.")


def compute_global_percentiles(scores: np.ndarray, reference_scores: np.ndarray | None) -> np.ndarray:
    if reference_scores is None or len(reference_scores) == 0:
        return pd.Series(scores).rank(pct=True).values * 100

    positions = np.searchsorted(reference_scores, scores, side="right")
    return positions / len(reference_scores) * 100


def assign_priority_from_percentile(percentile: float) -> str:
    if percentile >= 99:
        return "critical"
    if percentile >= 95:
        return "high"
    if percentile >= 80:
        return "medium"
    return "low"


def assign_probability_band(score: float) -> str:
    if score >= 0.80:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def build_risk_comment(row) -> str:
    score = float(row["score_illicit"])
    priority = str(row["priority_level"])

    if priority == "critical":
        return "Prioridad crítica: revisar inmediatamente por alta posición relativa frente al histórico de scores."
    if priority == "high":
        return "Prioridad alta: requiere revisión AML prioritaria."
    if priority == "medium":
        return "Prioridad media: revisar si existen señales adicionales de grafo, reglas o contexto externo."
    if score >= 0.50:
        return "Score moderado: monitoreo recomendado aunque no esté en percentiles superiores."
    return "Prioridad baja: no se prioriza salvo que existan señales externas adicionales."


st.title("Inferencia real con XGBoost")

st.caption(
    "Carga un archivo con las features completas del modelo para calcular nuevos scores de riesgo ilícito."
)

st.warning(
    """
Este módulo corresponde a inferencia local con el modelo XGBoost entrenado. 
No basta con cargar un txId: el archivo debe contener todas las variables requeridas por el modelo.
Las variables de grafo, como grados, deben venir previamente calculadas.
"""
)


schema = load_schema()
required_features = schema.get("required_features", [])

if not required_features:
    st.error("No se encontraron features requeridas en el esquema de inferencia.")
    st.stop()


st.subheader("1. Esquema requerido")

col_schema_1, col_schema_2, col_schema_3 = st.columns(3)

col_schema_1.metric("Modelo", schema.get("model_name", "XGBoost baseline"))
col_schema_2.metric("Features requeridas", f"{len(required_features):,}")
col_schema_3.metric("Columna objetivo", schema.get("target_output", "score_illicit"))

with st.expander("Ver columnas requeridas"):
    st.code("\n".join(required_features), language="text")


template_df = load_template()

st.subheader("2. Descargar plantilla")

col_download_1, col_download_2 = st.columns(2)

with col_download_1:
    st.download_button(
        label="Descargar plantilla CSV",
        data=dataframe_to_csv_bytes(template_df),
        file_name="plantilla_inferencia_xgboost.csv",
        mime="text/csv",
    )

with col_download_2:
    st.download_button(
        label="Descargar plantilla XLSX",
        data=dataframe_to_xlsx_bytes(template_df),
        file_name="plantilla_inferencia_xgboost.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


st.subheader("3. Cargar archivo de features")

uploaded_file = st.file_uploader(
    "Sube un archivo CSV o XLSX con todas las columnas requeridas",
    type=["csv", "xlsx"],
)

if uploaded_file is None:
    st.info("Carga un archivo para ejecutar inferencia.")
    st.stop()


try:
    input_df = read_uploaded_file(uploaded_file)
except Exception as error:
    st.error(f"No se pudo leer el archivo: {error}")
    st.stop()


st.markdown("### Vista previa del archivo cargado")
st.dataframe(input_df.head(10), width="stretch")


missing_features = [col for col in required_features if col not in input_df.columns]
extra_columns = [col for col in input_df.columns if col not in required_features + ["txId"]]

if missing_features:
    st.error(
        f"El archivo no contiene todas las columnas requeridas. "
        f"Faltan {len(missing_features)} columnas."
    )

    with st.expander("Ver columnas faltantes"):
        st.code("\n".join(missing_features), language="text")

    st.stop()


if extra_columns:
    st.info(
        f"El archivo contiene {len(extra_columns)} columnas extra. "
        "Serán ignoradas durante la inferencia."
    )


inference_df = input_df.copy()

for col in required_features:
    inference_df[col] = pd.to_numeric(inference_df[col], errors="coerce")


rows_with_missing = inference_df[required_features].isna().any(axis=1).sum()

if rows_with_missing > 0:
    st.error(
        f"Hay {rows_with_missing} filas con valores no numéricos o nulos en las features requeridas. "
        "Corrige el archivo antes de ejecutar inferencia."
    )
    st.stop()


st.subheader("4. Ejecutar inferencia")

threshold = st.slider(
    "Umbral para clase predicha",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
)

run_inference = st.button("Ejecutar inferencia XGBoost")

if not run_inference:
    st.stop()


try:
    model = load_model()
except Exception as error:
    st.error(
        f"No se pudo cargar el modelo XGBoost: {error}\n\n"
        "Este módulo funciona localmente si existe models/baseline_xgboost.pkl "
        "y el entorno tiene las dependencias necesarias."
    )
    st.stop()


X = inference_df[required_features].astype(float)

try:
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        raw_pred = model.predict(X)
        scores = np.asarray(raw_pred, dtype=float)
except Exception as error:
    st.error(f"Error durante la inferencia: {error}")
    st.stop()


reference_scores = load_reference_scores()
risk_percentiles = compute_global_percentiles(scores, reference_scores)

result_df = pd.DataFrame()

if "txId" in input_df.columns:
    result_df["txId"] = input_df["txId"].astype(str)
else:
    result_df["row_id"] = range(1, len(input_df) + 1)

result_df["score_illicit"] = scores
result_df["predicted_class"] = (scores >= threshold).astype(int)
result_df["predicted_label"] = result_df["predicted_class"].map(
    {
        0: "licit_predicted",
        1: "illicit_predicted",
    }
)

result_df["risk_percentile"] = risk_percentiles
result_df["priority_level"] = result_df["risk_percentile"].apply(assign_priority_from_percentile)
result_df["risk_level_probability"] = result_df["score_illicit"].apply(assign_probability_band)
result_df["risk_comment"] = result_df.apply(build_risk_comment, axis=1)

result_df = result_df.sort_values(
    ["priority_level", "score_illicit"],
    ascending=[True, False],
).reset_index(drop=True)

priority_order = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}

result_df["priority_order"] = result_df["priority_level"].map(priority_order)
result_df = result_df.sort_values(
    ["priority_order", "score_illicit"],
    ascending=[True, False],
).drop(columns=["priority_order"]).reset_index(drop=True)


st.subheader("5. Resumen de inferencia")

total_rows = len(result_df)
critical_count = int((result_df["priority_level"] == "critical").sum())
high_count = int((result_df["priority_level"] == "high").sum())
medium_count = int((result_df["priority_level"] == "medium").sum())
predicted_illicit_count = int((result_df["predicted_class"] == 1).sum())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Registros evaluados", f"{total_rows:,}")
col2.metric("Predichos ilícitos", f"{predicted_illicit_count:,}")
col3.metric("Critical + High", f"{critical_count + high_count:,}")
col4.metric("Score máximo", f"{result_df['score_illicit'].max():.4f}")

col5, col6, col7 = st.columns(3)

col5.metric("Critical", f"{critical_count:,}")
col6.metric("High", f"{high_count:,}")
col7.metric("Medium", f"{medium_count:,}")


st.subheader("6. Resultados")

st.dataframe(result_df, width="stretch")


st.subheader("7. Visualizaciones")

col_chart_1, col_chart_2 = st.columns(2)

with col_chart_1:
    fig_priority = px.bar(
        result_df["priority_level"].value_counts().reset_index(),
        x="priority_level",
        y="count",
        text="count",
        title="Distribución por prioridad",
    )
    st.plotly_chart(fig_priority, width="stretch")

with col_chart_2:
    fig_pred = px.bar(
        result_df["predicted_label"].value_counts().reset_index(),
        x="predicted_label",
        y="count",
        text="count",
        title="Distribución de clase predicha",
    )
    st.plotly_chart(fig_pred, width="stretch")


fig_score = px.histogram(
    result_df,
    x="score_illicit",
    nbins=30,
    title="Distribución de scores generados",
)

st.plotly_chart(fig_score, width="stretch")


st.subheader("8. Descargar resultados")

col_out_1, col_out_2 = st.columns(2)

with col_out_1:
    st.download_button(
        label="Descargar resultados CSV",
        data=dataframe_to_csv_bytes(result_df),
        file_name="resultados_inferencia_xgboost.csv",
        mime="text/csv",
    )

with col_out_2:
    st.download_button(
        label="Descargar resultados XLSX",
        data=dataframe_to_xlsx_bytes(result_df),
        file_name="resultados_inferencia_xgboost.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


st.divider()

st.subheader("Interpretación metodológica")

st.markdown(
    """
Este módulo ejecuta inferencia con el modelo XGBoost previamente entrenado. A diferencia del módulo de consulta por `txId`,
aquí el archivo cargado debe contener las variables completas utilizadas durante el entrenamiento. 

El resultado debe interpretarse como un score de apoyo para priorización AML. Para una aplicación real, sería necesario
garantizar que las features se calculen con el mismo pipeline usado en entrenamiento, validar el origen de los datos y
monitorear posibles cambios de distribución.
"""
)