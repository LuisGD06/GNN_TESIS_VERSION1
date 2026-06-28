from pathlib import Path
import json
import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "baseline_xgboost.pkl"

CANDIDATE_DATA_PATHS = [
    PROJECT_ROOT / "data" / "interim" / "elliptic" / "nodes_with_graph_features.parquet",
    PROJECT_ROOT / "data" / "interim" / "elliptic" / "nodes_interim.parquet",
    PROJECT_ROOT / "data" / "processed" / "elliptic" / "platform_predictions.parquet",
]

SCHEMA_OUTPUT_PATH = PROJECT_ROOT / "reports" / "metrics" / "inference_schema_xgboost.json"
TEMPLATE_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "elliptic" / "inference_template_xgboost.csv"


def default_feature_list() -> list[str]:
    return [f"f_{i}" for i in range(165)] + [
        "in_degree",
        "out_degree",
        "total_degree",
    ]


def get_model_features(model) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        if booster.feature_names:
            return list(booster.feature_names)

    if hasattr(model, "named_steps"):
        for _, step in reversed(model.named_steps.items()):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
            if hasattr(step, "get_booster"):
                booster = step.get_booster()
                if booster.feature_names:
                    return list(booster.feature_names)

    return default_feature_list()


def load_best_available_data() -> pd.DataFrame | None:
    for path in CANDIDATE_DATA_PATHS:
        if path.exists():
            print(f"Leyendo datos de referencia desde: {path}")
            return pd.read_parquet(path)

    return None


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo principal: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    feature_names = get_model_features(model)

    data_df = load_best_available_data()

    SCHEMA_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEMPLATE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    schema = {
        "model_name": "XGBoost baseline",
        "model_path": str(MODEL_PATH),
        "num_features": len(feature_names),
        "required_features": feature_names,
        "optional_columns": ["txId"],
        "target_output": "score_illicit",
        "prediction_type": "binary_classification",
        "positive_class": "illicit",
        "notes": [
            "El archivo de inferencia debe contener todas las columnas requeridas.",
            "Las columnas extra serán ignoradas.",
            "El modelo espera el mismo esquema de variables usado durante el entrenamiento.",
            "El módulo no reconstruye variables de grafo; estas deben venir calculadas en el archivo de entrada.",
        ],
    }

    with open(SCHEMA_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(schema, file, indent=2, ensure_ascii=False)

    if data_df is not None and all(col in data_df.columns for col in feature_names):
        template_cols = []

        if "txId" in data_df.columns:
            template_cols.append("txId")

        template_cols += feature_names

        template_df = data_df[template_cols].head(10).copy()
    else:
        template_data = {"txId": ["ejemplo_1", "ejemplo_2"]}

        for col in feature_names:
            template_data[col] = [0.0, 0.0]

        template_df = pd.DataFrame(template_data)

    template_df.to_csv(TEMPLATE_OUTPUT_PATH, index=False)

    print("\nNivel 9B preparado: esquema y plantilla de inferencia generados.")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Número de features requeridas: {len(feature_names)}")
    print(f"Schema: {SCHEMA_OUTPUT_PATH}")
    print(f"Template: {TEMPLATE_OUTPUT_PATH}")
    print("\nPrimeras columnas requeridas:")
    print(feature_names[:15])
    print("\nÚltimas columnas requeridas:")
    print(feature_names[-10:])


if __name__ == "__main__":
    main()