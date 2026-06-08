import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm


def safe_divide(a, b):
    if b == 0:
        return 0.0
    return float(a) / float(b)


def get_k_hop_nodes_fast(graph, undirected_graph, center_node, k=1, direction="both"):
    """
    Obtiene nodos k-hop evitando reconstruir el grafo no dirigido en cada iteración.
    """
    if center_node not in graph:
        return set()

    if direction == "out":
        lengths = nx.single_source_shortest_path_length(
            graph,
            center_node,
            cutoff=k
        )
        return set(lengths.keys())

    if direction == "in":
        reverse_graph = graph.reverse(copy=False)
        lengths = nx.single_source_shortest_path_length(
            reverse_graph,
            center_node,
            cutoff=k
        )
        return set(lengths.keys())

    if direction == "both":
        lengths = nx.single_source_shortest_path_length(
            undirected_graph,
            center_node,
            cutoff=k
        )
        return set(lengths.keys())

    raise ValueError("direction debe ser 'out', 'in' o 'both'.")


def compute_center_neighborhood_features_fast(
    graph,
    undirected_graph,
    labels_map,
    in_degree_map,
    out_degree_map,
    total_degree_map,
    center_node,
    k=1,
    direction="both",
):
    """
    Calcula features estructurales de un vecindario k-hop sin copiar subgrafos completos.
    """
    if center_node not in graph:
        return {
            "center_txId": center_node,
            "k": k,
            "direction": direction,
            "subgraph_num_nodes": 0,
            "subgraph_num_edges": 0,
            "subgraph_density": 0.0,
            "center_in_degree": 0,
            "center_out_degree": 0,
            "center_total_degree": 0,
            "num_licit_neighbors": 0,
            "num_illicit_neighbors": 0,
            "num_unknown_neighbors": 0,
            "ratio_licit_neighbors": 0.0,
            "ratio_illicit_neighbors": 0.0,
            "ratio_unknown_neighbors": 0.0,
            "avg_neighbor_total_degree": 0.0,
            "max_neighbor_total_degree": 0,
            "fan_in_out_ratio": 0.0,
        }

    sub_nodes = get_k_hop_nodes_fast(
        graph=graph,
        undirected_graph=undirected_graph,
        center_node=center_node,
        k=k,
        direction=direction
    )

    neighbor_nodes = [node for node in sub_nodes if node != center_node]

    n_nodes = len(sub_nodes)

    subgraph_view = graph.subgraph(sub_nodes)
    n_edges = subgraph_view.number_of_edges()

    center_in_degree = in_degree_map.get(center_node, 0)
    center_out_degree = out_degree_map.get(center_node, 0)
    center_total_degree = total_degree_map.get(center_node, 0)

    neighbor_labels = [
        labels_map.get(node, "unknown")
        for node in neighbor_nodes
    ]

    num_neighbors = len(neighbor_nodes)

    num_licit = sum(label == "licit" for label in neighbor_labels)
    num_illicit = sum(label == "illicit" for label in neighbor_labels)
    num_unknown = sum(label == "unknown" for label in neighbor_labels)

    neighbor_degrees = [
        total_degree_map.get(node, 0)
        for node in neighbor_nodes
    ]

    avg_neighbor_total_degree = (
        float(np.mean(neighbor_degrees))
        if neighbor_degrees
        else 0.0
    )

    max_neighbor_total_degree = (
        int(np.max(neighbor_degrees))
        if neighbor_degrees
        else 0
    )

    subgraph_density = (
        safe_divide(n_edges, n_nodes * (n_nodes - 1))
        if n_nodes > 1
        else 0.0
    )

    return {
        "center_txId": center_node,
        "k": k,
        "direction": direction,
        "subgraph_num_nodes": int(n_nodes),
        "subgraph_num_edges": int(n_edges),
        "subgraph_density": float(subgraph_density),
        "center_in_degree": int(center_in_degree),
        "center_out_degree": int(center_out_degree),
        "center_total_degree": int(center_total_degree),
        "num_licit_neighbors": int(num_licit),
        "num_illicit_neighbors": int(num_illicit),
        "num_unknown_neighbors": int(num_unknown),
        "ratio_licit_neighbors": safe_divide(num_licit, num_neighbors),
        "ratio_illicit_neighbors": safe_divide(num_illicit, num_neighbors),
        "ratio_unknown_neighbors": safe_divide(num_unknown, num_neighbors),
        "avg_neighbor_total_degree": avg_neighbor_total_degree,
        "max_neighbor_total_degree": max_neighbor_total_degree,
        "fan_in_out_ratio": safe_divide(center_in_degree, center_out_degree),
    }


def build_subgraph_feature_table(
    graph,
    nodes_df,
    center_nodes_df,
    k_values=(1, 2),
    direction="both",
    node_col="txId",
    label_col="label",
    show_progress=True,
):
    """
    Construye una tabla de features de subgrafos para nodos centro etiquetados.

    Usa el grafo completo como contexto, pero etiqueta cada subgrafo según el nodo central.
    """
    labels_map = dict(zip(nodes_df[node_col], nodes_df[label_col]))

    in_degree_map = dict(graph.in_degree())
    out_degree_map = dict(graph.out_degree())
    total_degree_map = dict(graph.degree())

    undirected_graph = graph.to_undirected(as_view=True)

    center_records = center_nodes_df[
        [node_col, label_col, "timestep"]
    ].to_dict(orient="records")

    iterator = center_records

    if show_progress:
        iterator = tqdm(center_records, desc="Extrayendo features de subgrafos")

    rows = []

    for row in iterator:
        center_node = row[node_col]
        center_label = row[label_col]
        center_timestep = row["timestep"]

        for k in k_values:
            features = compute_center_neighborhood_features_fast(
                graph=graph,
                undirected_graph=undirected_graph,
                labels_map=labels_map,
                in_degree_map=in_degree_map,
                out_degree_map=out_degree_map,
                total_degree_map=total_degree_map,
                center_node=center_node,
                k=k,
                direction=direction
            )

            features["label"] = center_label
            features["target"] = 1 if center_label == "illicit" else 0
            features["timestep"] = int(center_timestep)

            rows.append(features)

    return pd.DataFrame(rows)