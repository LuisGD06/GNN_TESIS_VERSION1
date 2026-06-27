import sys
from pathlib import Path
import json

import pandas as pd
import streamlit as st
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REPORTS_TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
REPORTS_METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

SUMMARY_PATH = REPORTS_METRICS_DIR / "temporal_error_analysis_summary.json"
METRICS_BY_SPLIT_PATH = REPORTS_TABLES_DIR / "temporal_error_metrics_by_split.csv"
CONFUSION_BY_SPLIT_PATH = REPORTS_TABLES_DIR / "temporal_error_confusion_by_split.csv"
RECALL_BY_SPLIT_PATH = REPORTS_TABLES_DIR / "temporal_error_recall_at_k_by_split.csv"
SCORE_BY_LABEL_SPLIT_PATH = REPORTS_TABLES_DIR / "temporal_error_score_by_label_split.csv"
FP_TEST_PATH = REPORTS_TABLES_DIR / "temporal_error_false_positives_test_top.csv"
FN_TEST_PATH = REPORTS_TABLES_DIR / "temporal_error_false_negatives_test_top.csv"
SPLIT_LABEL_DISTRIBUTION_PATH = REPORTS_TABLES_DIR / "temporal_split_label_distribution.csv"


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


st.set_page_config(page_title="Validación temporal", layout="wide")

st.title("Validación temporal del modelo")
st.caption(
    "Evaluación separada por train, validation y test para evitar interpretar métricas globales como desempeño de generalización."
)

if not SUMMARY_PATH.exists():
    st.error(
        "No se encontró temporal_error_analysis_summary.json. "
        "Ejecuta primero: python scripts/09_temporal_error_validation.py"
    )
    st.stop()

summary = load_json(SUMMARY_PATH)

metrics_df = load_csv(METRICS_BY_SPLIT_PATH)
confusion_df = load_csv(CONFUSION_BY_SPLIT_PATH)
recall_df = load_csv(RECALL_BY_SPLIT_PATH)
score_df = load_csv(SCORE_BY_LABEL_SPLIT_PATH)
fp_test_df = load_csv(FP_TEST_PATH)
fn_test_df = load_csv(FN_TEST_PATH)
split_label_df = load_csv(SPLIT_LABEL_DISTRIBUTION_PATH)

st.subheader("Definición de splits")

st.json(summary.get("split_definition", {}))

st.warning(
    "Para la tesis, las métricas principales de desempeño deben reportarse sobre el split test. "
    "Las métricas globales sobre todos los nodos etiquetados son descriptivas, no una medida final de generalización."
)

st.divider()

st.subheader("Métricas por split temporal")

st.dataframe(metrics_df, width="stretch")

fig_metrics = px.bar(
    metrics_df.melt(
        id_vars=["split"],
        value_vars=["precision", "recall", "f1", "accuracy"],
        var_name="metric",
        value_name="value",
    ),
    x="split",
    y="value",
    color="metric",
    barmode="group",
    title="Métricas por split temporal",
)

st.plotly_chart(fig_metrics, width="stretch")

st.divider()

st.subheader("Distribución de etiquetas por split")

st.dataframe(split_label_df, width="stretch")

fig_label_split = px.bar(
    split_label_df,
    x="split",
    y="count",
    color="label",
    barmode="group",
    title="Distribución de etiquetas por split",
)

st.plotly_chart(fig_label_split, width="stretch")

st.divider()

st.subheader("Tipos de error por split")

st.dataframe(confusion_df, width="stretch")

fig_confusion = px.bar(
    confusion_df,
    x="split",
    y="count",
    color="error_type",
    barmode="group",
    title="TP, TN, FP y FN por split",
)

st.plotly_chart(fig_confusion, width="stretch")

st.divider()

st.subheader("Recall@K por split")

st.dataframe(recall_df, width="stretch")

fig_recall = px.line(
    recall_df,
    x="k",
    y="recall_at_k",
    color="split",
    markers=True,
    title="Recall@K comparado por split",
)

st.plotly_chart(fig_recall, width="stretch")

st.info(
    "Recall@K permite evaluar qué proporción de ilícitos conocidos aparece dentro de las primeras K alertas priorizadas."
)

st.divider()

st.subheader("Score promedio por etiqueta y split")

st.dataframe(score_df, width="stretch")

fig_score = px.bar(
    score_df,
    x="split",
    y="mean",
    color="label",
    barmode="group",
    title="Score ilícito promedio por etiqueta y split",
)

st.plotly_chart(fig_score, width="stretch")

st.divider()

st.subheader("Falsos positivos del test")

st.markdown(
    """
Estos casos son transacciones etiquetadas como lícitas en el split test, pero priorizadas como riesgosas por el modelo.
En un entorno AML, pueden representar carga de revisión adicional o casos estructuralmente atípicos.
"""
)

st.dataframe(fp_test_df, width="stretch")

st.divider()

st.subheader("Falsos negativos del test")

st.markdown(
    """
Estos casos son los más críticos: transacciones ilícitas del split test que no fueron priorizadas correctamente.
Deben ser analizados para mejorar umbrales, reglas complementarias o señales estructurales.
"""
)

st.dataframe(fn_test_df, width="stretch")

st.divider()

st.subheader("Conclusión metodológica")

st.markdown(
    """
La validación temporal permite distinguir entre el comportamiento descriptivo del sistema sobre todos los nodos etiquetados
y el desempeño real de generalización sobre datos futuros. Esta separación es fundamental en problemas AML, porque las
estrategias de lavado, la conectividad del grafo y la proporción de ilícitos pueden cambiar a través del tiempo.

Por ello, el prototipo debe interpretarse como un sistema de priorización de alertas que combina score predictivo,
análisis estructural y revisión humana, no como un clasificador automático definitivo.
"""
)