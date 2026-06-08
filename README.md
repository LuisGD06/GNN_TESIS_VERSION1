# GNN_TESIS_VERSION1

Prototipo analítico para detección y priorización de transacciones ilícitas en Bitcoin usando el Elliptic Data Set, modelos clásicos, análisis de grafos, GNN y subgrafos.

## Objetivo

Diseñar y desarrollar un prototipo de plataforma analítica para apoyar la detección de transacciones ilícitas en Bitcoin mediante técnicas de ciencia de datos, aprendizaje automático y análisis de grafos.

## Metodología

El proyecto se desarrolló por niveles:

1. Análisis exploratorio del grafo.
2. Baselines clásicos.
3. Clasificación de nodos con GNN.
4. Análisis de subgrafos.
5. Baselines con features de subgrafos.
6. Exportación de artefactos para plataforma.
7. Dashboard AML en Streamlit.

## Resultado principal

El modelo principal seleccionado fue XGBoost baseline, debido a su mejor desempeño global según PR-AUC en el conjunto de prueba.

Las GNN y el análisis de subgrafos se conservaron como componentes exploratorios e interpretativos, ya que aportan contexto estructural sobre las transacciones y sus vecindarios.

## Ejecutar dashboard

```bash
conda activate gnn_tesis
streamlit run app/streamlit_app.py