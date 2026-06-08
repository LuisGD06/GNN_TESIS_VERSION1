import torch
from torch_geometric.data import Data


def build_node_index(nodes_df, node_col="txId"):
    """
    Crea un mapeo txId -> índice entero continuo.
    """
    node_ids = nodes_df[node_col].tolist()
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    idx_to_node = {idx: node_id for node_id, idx in node_to_idx.items()}

    return node_to_idx, idx_to_node


def build_edge_index(edges_df, node_to_idx, source_col="txId1", target_col="txId2"):
    """
    Construye edge_index para PyTorch Geometric.
    """
    sources = edges_df[source_col].map(node_to_idx)
    targets = edges_df[target_col].map(node_to_idx)

    valid_mask = sources.notna() & targets.notna()

    edge_index = torch.tensor(
        [
            sources[valid_mask].astype(int).to_numpy(),
            targets[valid_mask].astype(int).to_numpy(),
        ],
        dtype=torch.long,
    )

    return edge_index


def build_pyg_data(
    nodes_df,
    edges_df,
    feature_cols,
    target_col="target",
    split_col="split",
    node_col="txId",
):
    """
    Construye un objeto Data de PyTorch Geometric para clasificación de nodos.
    """
    node_to_idx, idx_to_node = build_node_index(nodes_df, node_col=node_col)

    x = torch.tensor(
        nodes_df[feature_cols].to_numpy(),
        dtype=torch.float,
    )

    y = torch.tensor(
        nodes_df[target_col].to_numpy(),
        dtype=torch.long,
    )

    edge_index = build_edge_index(edges_df, node_to_idx)

    train_mask = torch.tensor(
        (nodes_df[split_col] == "train").to_numpy(),
        dtype=torch.bool,
    )

    val_mask = torch.tensor(
        (nodes_df[split_col] == "val").to_numpy(),
        dtype=torch.bool,
    )

    test_mask = torch.tensor(
        (nodes_df[split_col] == "test").to_numpy(),
        dtype=torch.bool,
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    return data, node_to_idx, idx_to_node
