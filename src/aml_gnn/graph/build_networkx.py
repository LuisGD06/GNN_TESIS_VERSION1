import networkx as nx


def build_directed_graph(edges_df, source_col="txId1", target_col="txId2"):
    """
    Construye un grafo dirigido NetworkX a partir del edgelist del dataset Elliptic.
    """
    graph = nx.DiGraph()

    edges = edges_df[[source_col, target_col]].itertuples(index=False, name=None)
    graph.add_edges_from(edges)

    return graph
