# Arquitectura de plataforma

## Flujo general

CSV crudos -> validación -> preprocesamiento -> construcción de grafo -> inferencia -> score AML -> dashboard.

## Artefactos esperados

- predictions.parquet
- risk_scores.parquet
- subgraph_alerts.parquet
- metrics_summary.json
- graph_samples.json

## Módulos de plataforma

- Ingesta de datos.
- Motor de inferencia.
- Motor de scoring AML.
- Visualización de transacciones.
- Visualización de subgrafos.
- Reporte de métricas.
