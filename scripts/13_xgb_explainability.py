from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import joblib


warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "baseline_xgboost.pkl"
SCHEMA_PATH = PROJECT_ROOT / "reports" / "metrics" / "inference_schema_xgboost.json"
FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "elliptic" / "nodes_with_graph_features.parquet"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "elliptic" / "platform_predictions.parquet"

REPORTS_TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
REPORTS_METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

MODEL_IMPORTANCE_PATH = REPORTS_TABLES_DIR / "xgb_model_feature_importance.csv"
SHAP_GLOBAL_PATH = REPORTS_TABLES_DIR / "xgb_shap_global_importance.csv"
LOCAL_EXPLANATIONS_PATH = REPORTS_TABLES_DIR / "xgb_local_explanations_top_alerts.csv"
SUMMARY_PATH = REPORTS_METRICS_DIR / "xgb_explainability_summary.json"


def assign_split(timestep: int) -> str:
    if timestep <= 34:
        return "train"
    if 35 <= timestep <= 42:
        return "validation"
    return "test"


def load_required_features() -> list[str]:
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
            schema = json.load(file)
        return list(schema["required_features"])

    return [f"f_{i}" for i in range(165)] + [
        "in_degree",
        "out_degree",
        "total_degree",
    ]


def classify_feature_group(feature: str) -> str:
    if feature in ["in_degree", "out_degree", "total_degree"]:
        return "graph_degree_feature"
    if feature.startswith("f_"):
        return "elliptic_original_feature"
    return "other"


def get_model_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "get_booster"):
        booster = model.get_booster()
        score_dict = booster.get_score(importance_type="gain")
        importances = np.array(
            [score_dict.get(feature, 0.0) for feature in feature_names],
            dtype=float,
        )
    else:
        importances = np.zeros(len(feature_names), dtype=float)

    if len(importances) != len(feature_names):
        importances = np.resize(importances, len(feature_names))

    total = importances.sum()

    if total > 0:
        normalized = importances / total
    else:
        normalized = importances

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "model_importance": importances,
            "model_importance_normalized": normalized,
        }
    )

    importance_df["feature_group"] = importance_df["feature"].apply(classify_feature_group)

    importance_df = importance_df.sort_values(
        "model_importance",
        ascending=False,
    ).reset_index(drop=True)

    importance_df["rank_model_importance"] = importance_df.index + 1

    return importance_df


def compute_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    try:
        import shap
    except Exception as error:
        raise ImportError(
            "SHAP no está instalado. Ejecuta: python -m pip install shap"
        ) from error

    # Compatibilidad con algunas versiones de SHAP/Numpy.
    if not hasattr(np, "bool"):
        np.bool = np.bool_

    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X)

    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1]
    else:
        shap_values = shap_values_raw

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    return shap_values


def build_top_feature_text(feature_names: list[str], shap_row: np.ndarray, top_n: int = 5) -> tuple[str, str]:
    contribution_df = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": shap_row,
        }
    )

    positive_df = (
        contribution_df[contribution_df["shap_value"] > 0]
        .sort_values("shap_value", ascending=False)
        .head(top_n)
    )

    negative_df = (
        contribution_df[contribution_df["shap_value"] < 0]
        .sort_values("shap_value", ascending=True)
        .head(top_n)
    )

    positive_text = "; ".join(
        [
            f"{row.feature}: {row.shap_value:.5f}"
            for row in positive_df.itertuples()
        ]
    )

    negative_text = "; ".join(
        [
            f"{row.feature}: {row.shap_value:.5f}"
            for row in negative_df.itertuples()
        ]
    )

    return positive_text, negative_text


def main() -> None:
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

    if not FEATURE_DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el dataset con features: {FEATURE_DATA_PATH}")

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"No se encontró platform_predictions: {PREDICTIONS_PATH}")

    model = joblib.load(MODEL_PATH)
    feature_names = load_required_features()

    data_df = pd.read_parquet(FEATURE_DATA_PATH)
    predictions_df = pd.read_parquet(PREDICTIONS_PATH)

    missing_features = [
        feature for feature in feature_names
        if feature not in data_df.columns
    ]

    if missing_features:
        raise ValueError(f"Faltan features en data_df: {missing_features[:20]}")

    data_df["split"] = data_df["timestep"].apply(assign_split)

    # ==========================================================
    # 1. Importancia interna del modelo
    # ==========================================================

    model_importance_df = get_model_feature_importance(model, feature_names)
    model_importance_df.to_csv(MODEL_IMPORTANCE_PATH, index=False)

    # ==========================================================
    # 2. SHAP global sobre muestra de test
    # ==========================================================

    test_labeled_df = data_df[
        (data_df["split"] == "test")
        & (data_df["label"].isin(["licit", "illicit"]))
    ].copy()

    sample_size = min(2000, len(test_labeled_df))

    shap_sample_df = (
        test_labeled_df
        .sample(n=sample_size, random_state=42)
        .copy()
        if sample_size > 0
        else test_labeled_df.copy()
    )

    X_sample = shap_sample_df[feature_names].astype(float)

    shap_values = compute_shap_values(model, X_sample)

    shap_global_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
        }
    )

    shap_global_df["feature_group"] = shap_global_df["feature"].apply(classify_feature_group)

    shap_global_df = shap_global_df.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    shap_global_df["rank_shap_importance"] = shap_global_df.index + 1

    shap_global_df.to_csv(SHAP_GLOBAL_PATH, index=False)

    # ==========================================================
    # 3. Explicaciones locales para top alertas
    # ==========================================================

    data_df["txId_str"] = data_df["txId"].astype(str)
    predictions_df["txId_str"] = predictions_df["txId"].astype(str)

    top_alerts_df = (
        predictions_df
        .sort_values("risk_rank")
        .head(50)
        .copy()
    )

    top_alert_features_df = top_alerts_df[["txId_str"]].merge(
        data_df[["txId_str"] + feature_names],
        on="txId_str",
        how="inner",
    )

    local_explanations_rows = []

    if not top_alert_features_df.empty:
        X_top = top_alert_features_df[feature_names].astype(float)
        shap_top_values = compute_shap_values(model, X_top)

        top_alert_meta_df = top_alert_features_df[["txId_str"]].merge(
            top_alerts_df,
            on="txId_str",
            how="left",
        )

        for idx, row in top_alert_meta_df.iterrows():
            positive_text, negative_text = build_top_feature_text(
                feature_names,
                shap_top_values[idx],
                top_n=5,
            )

            local_explanations_rows.append(
                {
                    "txId": row.get("txId"),
                    "label": row.get("label"),
                    "timestep": row.get("timestep"),
                    "score_illicit": row.get("score_illicit"),
                    "priority_level": row.get("priority_level"),
                    "risk_rank": row.get("risk_rank"),
                    "top_positive_features": positive_text,
                    "top_negative_features": negative_text,
                }
            )

    local_explanations_df = pd.DataFrame(local_explanations_rows)
    local_explanations_df.to_csv(LOCAL_EXPLANATIONS_PATH, index=False)

    # ==========================================================
    # 4. Resumen
    # ==========================================================

    top_model_features = model_importance_df.head(10)["feature"].tolist()
    top_shap_features = shap_global_df.head(10)["feature"].tolist()

    summary = {
        "model": "XGBoost baseline",
        "num_required_features": len(feature_names),
        "shap_sample": {
            "split": "test",
            "sample_size": int(sample_size),
            "source": str(FEATURE_DATA_PATH),
        },
        "outputs": {
            "model_feature_importance": str(MODEL_IMPORTANCE_PATH),
            "shap_global_importance": str(SHAP_GLOBAL_PATH),
            "local_explanations_top_alerts": str(LOCAL_EXPLANATIONS_PATH),
        },
        "top_10_model_importance_features": top_model_features,
        "top_10_shap_features": top_shap_features,
        "feature_group_counts_top_20_shap": shap_global_df.head(20)["feature_group"].value_counts().to_dict(),
        "methodological_note": (
            "La importancia interna de XGBoost y SHAP se utilizan para interpretar el modelo principal. "
            "SHAP se calcula sobre una muestra del split test para explicar la contribución promedio de las variables "
            "en datos de generalización."
        ),
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\nNivel 13B completado: explicabilidad XGBoost generada.")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Features: {len(feature_names)}")
    print(f"Muestra SHAP test: {sample_size:,}")
    print("\nArchivos generados:")
    print(MODEL_IMPORTANCE_PATH)
    print(SHAP_GLOBAL_PATH)
    print(LOCAL_EXPLANATIONS_PATH)
    print(SUMMARY_PATH)

    print("\nTop 10 SHAP:")
    print(shap_global_df.head(10)[["feature", "mean_abs_shap", "feature_group"]])


if __name__ == "__main__":
    main()