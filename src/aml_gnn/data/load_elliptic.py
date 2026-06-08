from pathlib import Path
import pandas as pd


def load_raw_elliptic(data_dir: str | Path):
    """
    Carga los tres archivos principales del Elliptic Data Set.
    """
    data_dir = Path(data_dir)

    features_path = data_dir / "elliptic_txs_features.csv"
    classes_path = data_dir / "elliptic_txs_classes.csv"
    edges_path = data_dir / "elliptic_txs_edgelist.csv"

    missing = [
        path.name
        for path in [features_path, classes_path, edges_path]
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"Faltan archivos del dataset en {data_dir}: {missing}"
        )

    features = pd.read_csv(features_path, header=None)
    classes = pd.read_csv(classes_path)
    edges = pd.read_csv(edges_path)

    return features, classes, edges
