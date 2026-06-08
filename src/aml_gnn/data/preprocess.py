import pandas as pd


def preprocess_elliptic_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesa la matriz de características del dataset Elliptic.

    En el dataset original:
    - columna 0: txId
    - columna 1: timestep
    - columnas restantes: variables numéricas.
    """
    df = features.copy()

    df = df.rename(columns={0: "txId", 1: "timestep"})

    feature_cols = [
        col for col in df.columns
        if col not in ["txId", "timestep"]
    ]

    renamed_features = {
        col: f"f_{i}"
        for i, col in enumerate(feature_cols)
    }

    df = df.rename(columns=renamed_features)

    return df


def preprocess_elliptic_classes(classes: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza las etiquetas:
    - 1: illicit
    - 2: licit
    - unknown: unknown
    """
    df = classes.copy()

    if "class" not in df.columns:
        raise ValueError("El archivo de clases debe contener la columna 'class'.")

    df["label"] = df["class"].replace({
        "1": "illicit",
        "2": "licit",
        1: "illicit",
        2: "licit",
        "unknown": "unknown"
    })

    return df


def merge_features_classes(features_df: pd.DataFrame, classes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Une features y etiquetas usando txId.
    """
    return features_df.merge(
        classes_df[["txId", "label"]],
        on="txId",
        how="left"
    )
