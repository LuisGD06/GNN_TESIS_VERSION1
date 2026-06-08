import streamlit as st

from app.utils.data_loader import load_export_metadata, load_metrics_summary


st.set_page_config(page_title="Metodología", layout="wide")

st.title("Documentación metodológica")

metadata = load_export_metadata()
metrics_summary = load_metrics_summary()

st.subheader("Resumen del flujo metodológico")

st.markdown(
    """
El proyecto siguió una ruta por niveles:

1. **Nivel 1:** análisis exploratorio del grafo.
2. **Nivel 2:** baselines clásicos.
3. **Nivel 3:** clasificación de nodos con GNN.
4. **Nivel 4:** análisis de subgrafos.
5. **Nivel 5:** baselines con features de subgrafos.
6. **Nivel 6:** exportación de resultados para plataforma.
7. **Nivel 7:** prototipo de dashboard AML.
"""
)

st.subheader("Decisión del modelo principal")

if metrics_summary:
    primary = metrics_summary.get("primary_model", {})

    st.markdown(
        f"""
**Modelo seleccionado:** `{primary.get("name", "N/A")}`

**Criterio de selección:** mayor PR-AUC en el conjunto de prueba.

**PR-AUC:** `{primary.get("test_pr_auc", 0):.4f}`  
**ROC-AUC:** `{primary.get("test_roc_auc", 0):.4f}`  
**Precision:** `{primary.get("test_precision", 0):.4f}`  
**Recall:** `{primary.get("test_recall", 0):.4f}`  
**F1:** `{primary.get("test_f1", 0):.4f}`
"""
    )

st.subheader("Advertencias de interpretación")

st.warning(
    """
Los scores generados para nodos `unknown` no representan etiquetas confirmadas.
Deben interpretarse como priorización de riesgo para revisión analítica.
"""
)

st.info(
    """
El análisis de grafos y subgrafos aporta contexto estructural e interpretabilidad,
aunque el modelo predictivo principal seleccionado sea XGBoost.
"""
)

if metadata:
    st.subheader("Metadata de exportación")

    st.json(metadata)