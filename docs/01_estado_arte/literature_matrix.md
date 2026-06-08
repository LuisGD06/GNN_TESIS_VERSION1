# Estado del arte: matriz de literatura

| Línea | Paper / recurso | Aporte principal | Dificultad identificada | Uso en la tesis |
|---|---|---|---|---|
| Dataset base | Elliptic AML-GCN | Dataset temporal Bitcoin y comparación LR/RF/MLP/GCN | RF fuerte; GNN no siempre supera baseline | Justificar baseline clásico y GCN |
| GNN base | GCN | Propagación por vecindarios | Riesgo de suavizado excesivo | Baseline GNN |
| Escalabilidad | GraphSAGE | Agregación inductiva de vecinos | Nuevos nodos y grafos grandes | Modelo GNN principal |
| Atención | GAT | Pesos diferenciados por vecino | Mayor costo e interpretación cuidadosa | Comparación adicional |
| Self-supervised | Inspection-L | DGI + GIN + RF | Pocos labels y muchos unknown | Mejora avanzada |
| Temporal | Graph-Based LSTM / Temporal-GCN | Combina LSTM y GCN | Importancia del tiempo | Split temporal y análisis por timestep |
| Heterogéneo | Elliptic++ | Transacciones + direcciones | Heterogeneidad de actores | Trabajo futuro o ampliación |
| Subgrafos | Elliptic2 | AML como clasificación de subgrafos | Escalabilidad y subgrafos etiquetados | Justificar análisis de subgrafos |
| Subgraph learning | SubGNN / GLASS | Representación formal de subgrafos | Topología interna y conectividad externa | Sustento para GIN + pooling |
