import sys
from pathlib import Path

import streamlit as st
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.data_loader import (
    load_error_analysis_summary,
    load_error_confusion_summary,
    load_error_false_positives,
    load_error_false_negatives,
    load_error_score_by_label,
    load_error_score_by_timestep,
    load_error_priority_by_label,
    load_error_recall_at_k,
)


st.set_page_config(page_title="Análisis de errores", layout="wide")

st.title("Análisis de errores y validación AML")
st.caption(
    "Evaluación crítica del modelo principal: falsos positivos, falsos negativos, ranking de alertas y comportamiento temporal."
)

summary = load_error_analysis_summary()
confusion_df = load_error_confusion_summary()
false_positives_df = load_error_false_positives()
false_negatives_df = load_error_false_negatives()
score_by_label_df = load_error_score_by_label()
score_by_timestep_df = load_error_score_by_timestep()
priority_by_label_df = load_error_priority_by_label()
recall_at_k_df = load_error_recall_at_k()

if summary is None or confusion_df is None:
    st.error(
        "No se encontraron los archivos de análisis de errores. "
        "Ejecuta primero: python scripts/08_error_analysis.py"
    )
    st.stop()

st.subheader("Resumen general")

dataset_info = summary.get("dataset", {})
metrics = summary.get("metrics_from_platform_predictions", {})
confusion_counts = summary.get("confusion_matrix_counts", {})
unknown_summary = summary.get("unknown_risk_summary", {})

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total predicciones", f"{dataset_info.get('total_predictions', 0):,}")
col2.metric("Nodos evaluables", f"{dataset_info.get('total_labeled', 0):,}")
col3.metric("Nodos unknown", f"{dataset_info.get('total_unknown', 0):,}")
col4.metric("F1", f"{metrics.get('f1', 0):.4f}")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Precision", f"{metrics.get('precision', 0):.4f}")
col2.metric("Recall", f"{metrics.get('recall', 0):.4f}")
col3.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
col4.metric("Unknown critical", f"{unknown_summary.get('unknown_critical_count', 0):,}")

st.divider()

st.subheader("Matriz de errores")

st.dataframe(confusion_df, width="stretch")

fig_confusion = px.bar(
    confusion_df,
    x="error_type",
    y="count",
    text="count",
    title="Conteo por tipo de resultado",
)

fig_confusion.update_layout(xaxis_tickangle=-30)

st.plotly_chart(fig_confusion, width="stretch")

st.markdown(
    """
**Interpretación:**  
En un contexto AML, los falsos negativos son especialmente críticos porque representan transacciones ilícitas no detectadas.
Los falsos positivos generan carga operativa, pero pueden ser aceptables si la política de negocio prioriza reducir el riesgo de omisión.
"""
)

st.divider()

st.subheader("Recall@K")

if recall_at_k_df is not None:
    st.dataframe(recall_at_k_df, width="stretch")

    fig_recall = px.line(
        recall_at_k_df,
        x="k",
        y="recall_at_k",
        markers=True,
        title="Recall acumulado en el ranking de alertas",
    )

    st.plotly_chart(fig_recall, width="stretch")

    st.info(
        "Recall@K indica qué proporción de transacciones ilícitas conocidas aparece dentro de las primeras K alertas ordenadas por score."
    )

st.divider()

st.subheader("Distribución de score por etiqueta")

if score_by_label_df is not None:
    st.dataframe(score_by_label_df, width="stretch")

    fig_score_label = px.bar(
        score_by_label_df,
        x="label",
        y="mean",
        text="count",
        title="Score ilícito promedio por etiqueta",
    )

    st.plotly_chart(fig_score_label, width="stretch")

st.divider()

st.subheader("Score promedio por timestep")

if score_by_timestep_df is not None:
    label_options = sorted(score_by_timestep_df["label"].dropna().unique())

    selected_labels = st.multiselect(
        "Filtrar etiquetas",
        options=label_options,
        default=label_options,
    )

    filtered_timestep_df = score_by_timestep_df[
        score_by_timestep_df["label"].isin(selected_labels)
    ].copy()

    fig_timestep = px.line(
        filtered_timestep_df,
        x="timestep",
        y="mean_score",
        color="label",
        markers=True,
        title="Evolución temporal del score promedio por etiqueta",
    )

    st.plotly_chart(fig_timestep, width="stretch")

    st.dataframe(filtered_timestep_df, width="stretch")

st.divider()

st.subheader("Prioridad por etiqueta")

if priority_by_label_df is not None:
    st.dataframe(priority_by_label_df, width="stretch")

    fig_priority = px.bar(
        priority_by_label_df,
        x="priority_level",
        y="count",
        color="label",
        barmode="group",
        title="Distribución de prioridad por etiqueta",
    )

    st.plotly_chart(fig_priority, width="stretch")

st.divider()

st.subheader("Falsos positivos: lícitas priorizadas como riesgo")

if false_positives_df is not None:
    st.markdown(
        """
Los falsos positivos son transacciones etiquetadas como lícitas que recibieron una predicción riesgosa.
En una plataforma AML real, estos casos pueden generar revisión manual innecesaria, pero también sirven para detectar patrones atípicos.
"""
    )

    st.dataframe(false_positives_df, width="stretch")

st.divider()

st.subheader("Falsos negativos: ilícitas no detectadas")

if false_negatives_df is not None:
    st.markdown(
        """
Los falsos negativos son los casos más delicados en AML: transacciones ilícitas que el modelo no priorizó correctamente.
Estos casos deben analizarse para mejorar umbrales, features, reglas complementarias o revisión estructural.
"""
    )

    st.dataframe(false_negatives_df, width="stretch")

st.divider()

st.subheader("Conclusión analítica")

st.warning(
    """
El análisis de errores muestra que el modelo no debe interpretarse como un sistema automático de decisión final.
Su mejor uso dentro del prototipo es como motor de priorización de alertas, complementado con análisis estructural de grafos,
subgrafos y revisión humana.
"""
)