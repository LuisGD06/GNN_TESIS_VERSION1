import pandas as pd


def add_degree_features(nodes_df: pd.DataFrame, graph, node_col: str = "txId") -> pd.DataFrame:
    """
    Agrega grado de entrada, salida y total a la tabla de nodos.
    """
    df = nodes_df.copy()

    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    total_degree = dict(graph.degree())

    df["in_degree"] = df[node_col].map(in_degree).fillna(0).astype(int)
    df["out_degree"] = df[node_col].map(out_degree).fillna(0).astype(int)
    df["total_degree"] = df[node_col].map(total_degree).fillna(0).astype(int)

    return df
