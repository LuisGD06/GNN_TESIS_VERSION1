import networkx as nx
import numpy as np


def graph_basic_stats(graph: nx.DiGraph) -> dict:
    """
    Calcula estadísticas básicas del grafo dirigido.
    """
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    in_degrees = [degree for _, degree in graph.in_degree()]
    out_degrees = [degree for _, degree in graph.out_degree()]
    total_degrees = [degree for _, degree in graph.degree()]

    stats = {
        "num_nodes": n_nodes,
        "num_edges": n_edges,
        "density": nx.density(graph),
        "is_directed": graph.is_directed(),
        "avg_in_degree": float(np.mean(in_degrees)) if in_degrees else 0,
        "avg_out_degree": float(np.mean(out_degrees)) if out_degrees else 0,
        "avg_total_degree": float(np.mean(total_degrees)) if total_degrees else 0,
        "max_in_degree": int(np.max(in_degrees)) if in_degrees else 0,
        "max_out_degree": int(np.max(out_degrees)) if out_degrees else 0,
        "max_total_degree": int(np.max(total_degrees)) if total_degrees else 0,
        "num_weakly_connected_components": nx.number_weakly_connected_components(graph),
        "num_strongly_connected_components": nx.number_strongly_connected_components(graph),
    }

    return stats


def top_degree_nodes(graph: nx.DiGraph, degree_type="total", top_n=10):
    """
    Devuelve los nodos con mayor grado.

    degree_type:
    - total
    - in
    - out
    """
    if degree_type == "in":
        degree_view = graph.in_degree()
    elif degree_type == "out":
        degree_view = graph.out_degree()
    elif degree_type == "total":
        degree_view = graph.degree()
    else:
        raise ValueError("degree_type debe ser 'total', 'in' u 'out'.")

    degree_items = sorted(
        degree_view,
        key=lambda item: item[1],
        reverse=True
    )

    return degree_items[:top_n]
