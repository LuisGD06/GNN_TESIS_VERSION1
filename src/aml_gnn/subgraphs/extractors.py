import networkx as nx


def get_k_hop_nodes(graph, center_node, k=1, direction="both"):
    """
    Obtiene nodos dentro de k saltos alrededor de un nodo central.

    direction:
    - 'out': vecinos alcanzables siguiendo dirección de aristas.
    - 'in': vecinos que llegan al nodo central.
    - 'both': vecindario ignorando dirección.
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
        undirected_graph = graph.to_undirected(as_view=True)
        lengths = nx.single_source_shortest_path_length(
            undirected_graph,
            center_node,
            cutoff=k
        )
        return set(lengths.keys())

    raise ValueError("direction debe ser 'out', 'in' o 'both'.")


def extract_k_hop_subgraph(graph, center_node, k=1, direction="both"):
    """
    Extrae el subgrafo k-hop alrededor de un nodo central.
    """
    nodes = get_k_hop_nodes(
        graph=graph,
        center_node=center_node,
        k=k,
        direction=direction
    )

    return graph.subgraph(nodes).copy()
