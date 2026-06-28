import sys
from pathlib import Path
import json

import pandas as pd
import streamlit as st
import plotly.express as px


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "tables" / "xgb_model_feature_importance.csv"
SHAP_GLOBAL_PATH = PROJECT_ROOT / "reports" / "tables" / "xgb_shap_global_importance.csv"
LOCAL_EXPLANATIONS_PATH = PROJECT_ROOT / "reports" / "tables" / "xgb_local_explanations_top_alerts.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "metrics" / "xgb_explainability_summary.json"


st.set_page_config(page_title="Explicabilidad XGBoost", layout="wide")


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


st.title("Explicabilidad del modelo XGBoost")

st.caption(
    "Análisis de importancia de variables y explicaciones locales para el modelo principal del prototipo AML."
)

required_paths = [
    MODEL_IMPORTANCE_PATH,
    SHAP_GLOBAL_PATH,
    LOCAL_EXPLANATIONS_PATH,
    SUMMARY_PATH,
]

missing_paths = [
    str(path)
    for path in required_paths
    if not path.exists()
]

if missing_paths:
    st.error(
        "Faltan artefactos de explicabilidad. "
        "Ejecuta primero: python scripts/13_xgb_explainability.py"
    )

    with st.expander("Archivos faltantes"):
        st.code("\n".join(missing_paths), language="text")

    st.stop()


summary = load_json(SUMMARY_PATH)
model_importance_df = load_csv(MODEL_IMPORTANCE_PATH)
shap_global_df = load_csv(SHAP_GLOBAL_PATH)
local_explanations_df = load_csv(LOCAL_EXPLANATIONS_PATH)


st.info(
    """
La explicabilidad permite interpretar qué variables influyen más en el score del modelo.
Se muestran dos enfoques: importancia interna de XGBoost y SHAP global calculado sobre una muestra del split test.
"""
)

st.subheader("Resumen del análisis")

col1, col2, col3 = st.columns(3)

col1.metric("Modelo", summary.get("model", "XGBoost baseline"))
col2.metric("Features evaluadas", f"{summary.get('num_required_features', 0):,}")
col3.metric("Muestra SHAP", f"{summary.get('shap_sample', {}).get('sample_size', 0):,}")

st.markdown("### Top 10 variables por SHAP")

top_shap_features = summary.get("top_10_shap_features", [])

if top_shap_features:
    st.code("\n".join(top_shap_features), language="text")

st.divider()

st.subheader("Importancia global SHAP")

top_n = st.slider(
    "Número de variables a mostrar",
    min_value=10,
    max_value=50,
    value=20,
    step=5,
)

top_shap_df = shap_global_df.head(top_n).copy()

fig_shap = px.bar(
    top_shap_df.sort_values("mean_abs_shap", ascending=True),
    x="mean_abs_shap",
    y="feature",
    color="feature_group",
    orientation="h",
    title=f"Top {top_n} variables por importancia SHAP",
)

st.plotly_chart(fig_shap, width="stretch")

st.dataframe(top_shap_df, width="stretch")

st.markdown(
    """
**Interpretación:**  
Una mayor media del valor absoluto SHAP indica que la variable tiene mayor influencia promedio en la predicción del modelo.
Este resultado no indica necesariamente causalidad; indica contribución predictiva dentro del modelo entrenado.
"""
)

st.divider()

st.subheader("Importancia interna del modelo XGBoost")

top_model_df = model_importance_df.head(top_n).copy()

fig_model = px.bar(
    top_model_df.sort_values("model_importance_normalized", ascending=True),
    x="model_importance_normalized",
    y="feature",
    color="feature_group",
    orientation="h",
    title=f"Top {top_n} variables por importancia interna XGBoost",
)

st.plotly_chart(fig_model, width="stretch")

st.dataframe(top_model_df, width="stretch")

st.markdown(
    """
La importancia interna del modelo muestra qué variables fueron más utilizadas por XGBoost para construir divisiones.
Puede diferir de SHAP, porque SHAP mide contribución promedio a las predicciones.
"""
)

st.divider()

st.subheader("Comparación entre importancia SHAP e importancia interna")

comparison_df = shap_global_df.merge(
    model_importance_df[
        [
            "feature",
            "model_importance_normalized",
            "rank_model_importance",
        ]
    ],
    on="feature",
    how="left",
)

comparison_df = comparison_df[
    [
        "feature",
        "feature_group",
        "rank_shap_importance",
        "mean_abs_shap",
        "rank_model_importance",
        "model_importance_normalized",
    ]
].copy()

st.dataframe(comparison_df.head(50), width="stretch")

fig_scatter = px.scatter(
    comparison_df,
    x="model_importance_normalized",
    y="mean_abs_shap",
    color="feature_group",
    hover_data=["feature", "rank_shap_importance", "rank_model_importance"],
    title="Comparación de importancia interna vs SHAP",
)

st.plotly_chart(fig_scatter, width="stretch")

st.divider()

st.subheader("Explicaciones locales de alertas principales")

st.markdown(
    """
Las explicaciones locales muestran qué variables empujaron el score hacia mayor riesgo y cuáles lo redujeron para transacciones específicas.
"""
)

st.dataframe(local_explanations_df, width="stretch")

selected_txid = st.selectbox(
    "Selecciona una alerta para revisar explicación local",
    options=local_explanations_df["txId"].astype(str).tolist(),
)

selected_row = local_explanations_df[
    local_explanations_df["txId"].astype(str) == selected_txid
].iloc[0]

col_a, col_b, col_c, col_d = st.columns(4)

col_a.metric("txId", str(selected_row.get("txId")))
col_b.metric("Label", str(selected_row.get("label")))
col_c.metric("Score ilícito", f"{float(selected_row.get('score_illicit', 0)):.4f}")
col_d.metric("Prioridad", str(selected_row.get("priority_level")))

st.markdown("### Variables que aumentan el score")

st.code(str(selected_row.get("top_positive_features", "")), language="text")

st.markdown("### Variables que reducen el score")

st.code(str(selected_row.get("top_negative_features", "")), language="text")

st.divider()

st.subheader("Conclusión metodológica")

st.markdown(
    """
El análisis de explicabilidad permite complementar las métricas de evaluación con una lectura interpretativa del modelo.
En el contexto AML, esto es importante porque los analistas no solo requieren un score, sino también indicios sobre qué señales
contribuyeron a priorizar una transacción. Sin embargo, estas explicaciones deben interpretarse como evidencia del comportamiento
del modelo y no como prueba causal de actividad ilícita.
"""
)