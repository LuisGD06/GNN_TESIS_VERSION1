def validate_elliptic_shapes(features, classes, edges) -> dict:
    """
    Valida dimensiones y columnas básicas del dataset Elliptic.
    """
    return {
        "features_shape": features.shape,
        "classes_shape": classes.shape,
        "edges_shape": edges.shape,
        "features_columns": list(features.columns),
        "classes_columns": list(classes.columns),
        "edges_columns": list(edges.columns),
    }
