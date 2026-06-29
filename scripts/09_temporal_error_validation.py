from pathlib import Path
import json
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "elliptic" / "platform_predictions.parquet"

REPORTS_TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
REPORTS_METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)


def assign_split(timestep: int) -> str:
    """
    Split temporal usado en los experimentos principales del proyecto.

    train: timesteps 1-34
    validation: timesteps 35-41
    test: timesteps 42-49
    """
    if timestep <= 34:
        return "train"
    if 35 <= timestep <= 41:
        return "validation"
    return "test"


def normalize_predicted_class(value):
    """
    Convierte predicted_class a formato binario:
    1 = illicit
    0 = licit
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, str):
        value_lower = value.strip().lower()

        if value_lower in ["1", "illicit", "fraud", "risk", "riesgo"]:
            return 1

        if value_lower in ["0", "licit", "legit", "normal", "no_risk"]:
            return 0

    try:
        return int(value)
    except Exception:
        return np.nan


def compute_binary_metrics(df: pd.DataFrame) -> dict:
    tp = int(((df["y_true"] == 1) & (df["y_pred"] == 1)).sum())
    tn = int(((df["y_true"] == 0) & (df["y_pred"] == 0)).sum())
    fp = int(((df["y_true"] == 0) & (df["y_pred"] == 1)).sum())
    fn = int(((df["y_true"] == 1) & (df["y_pred"] == 0)).sum())

    total = len(df)

    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "total": int(total),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def recall_at_k(df: pd.DataFrame, k: int) -> float:
    positives = int((df["y_true"] == 1).sum())

    if positives == 0:
        return 0.0

    top_k = df.sort_values("score_illicit", ascending=False).head(k)
    found = int((top_k["y_true"] == 1).sum())

    return float(found / positives)


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"No se encontró: {PREDICTIONS_PATH}")

    predictions_df = pd.read_parquet(PREDICTIONS_PATH)

    required_columns = [
        "txId",
        "timestep",
        "label",
        "score_illicit",
        "predicted_class",
        "priority_level",
        "risk_rank",
    ]

    missing_columns = [col for col in required_columns if col not in predictions_df.columns]

    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")

    df = predictions_df.copy()
    df["split"] = df["timestep"].apply(assign_split)

    labeled_df = df[df["label"].isin(["licit", "illicit"])].copy()

    labeled_df["y_true"] = labeled_df["label"].map(
        {
            "licit": 0,
            "illicit": 1,
        }
    )

    labeled_df["y_pred"] = labeled_df["predicted_class"].apply(normalize_predicted_class)

    if labeled_df["y_pred"].isna().any():
        labeled_df["y_pred"] = (labeled_df["score_illicit"] >= 0.5).astype(int)

    labeled_df["y_pred"] = labeled_df["y_pred"].astype(int)

    labeled_df["error_type"] = np.select(
        [
            (labeled_df["y_true"] == 1) & (labeled_df["y_pred"] == 1),
            (labeled_df["y_true"] == 0) & (labeled_df["y_pred"] == 0),
            (labeled_df["y_true"] == 0) & (labeled_df["y_pred"] == 1),
            (labeled_df["y_true"] == 1) & (labeled_df["y_pred"] == 0),
        ],
        [
            "TP_illicit_detected",
            "TN_licit_correct",
            "FP_licit_alerted",
            "FN_illicit_missed",
        ],
        default="unknown",
    )

    # ==========================================================
    # 1. Conteo de etiquetas por split
    # ==========================================================

    split_label_distribution = (
        df.groupby(["split", "label"])
        .size()
        .reset_index(name="count")
    )

    split_label_distribution.to_csv(
        REPORTS_TABLES_DIR / "temporal_split_label_distribution.csv",
        index=False,
    )

    # ==========================================================
    # 2. Matriz de errores por split
    # ==========================================================

    confusion_by_split = (
        labeled_df.groupby(["split", "error_type"])
        .size()
        .reset_index(name="count")
    )

    confusion_by_split["split_total"] = confusion_by_split.groupby("split")["count"].transform("sum")
    confusion_by_split["percentage"] = confusion_by_split["count"] / confusion_by_split["split_total"]

    confusion_by_split.to_csv(
        REPORTS_TABLES_DIR / "temporal_error_confusion_by_split.csv",
        index=False,
    )

    # ==========================================================
    # 3. Métricas por split
    # ==========================================================

    metrics_rows = []

    for split_name, split_df in labeled_df.groupby("split"):
        metrics = compute_binary_metrics(split_df)
        metrics["split"] = split_name
        metrics_rows.append(metrics)

    metrics_by_split = pd.DataFrame(metrics_rows)

    preferred_order = ["train", "validation", "test"]
    metrics_by_split["split"] = pd.Categorical(
        metrics_by_split["split"],
        categories=preferred_order,
        ordered=True,
    )

    metrics_by_split = metrics_by_split.sort_values("split")

    metrics_by_split.to_csv(
        REPORTS_TABLES_DIR / "temporal_error_metrics_by_split.csv",
        index=False,
    )

    # ==========================================================
    # 4. Recall@K por split
    # ==========================================================

    k_values = [50, 100, 200, 500, 1000, 2000, 5000]
    recall_rows = []

    for split_name, split_df in labeled_df.groupby("split"):
        for k in k_values:
            recall_rows.append(
                {
                    "split": split_name,
                    "k": k,
                    "recall_at_k": recall_at_k(split_df, k),
                    "num_records_split": int(len(split_df)),
                    "num_illicit_split": int((split_df["y_true"] == 1).sum()),
                }
            )

    recall_by_split = pd.DataFrame(recall_rows)

    recall_by_split["split"] = pd.Categorical(
        recall_by_split["split"],
        categories=preferred_order,
        ordered=True,
    )

    recall_by_split = recall_by_split.sort_values(["split", "k"])

    recall_by_split.to_csv(
        REPORTS_TABLES_DIR / "temporal_error_recall_at_k_by_split.csv",
        index=False,
    )

    # ==========================================================
    # 5. Score por etiqueta y split
    # ==========================================================

    score_by_label_split = (
        df.groupby(["split", "label"])["score_illicit"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )

    score_by_label_split["split"] = pd.Categorical(
        score_by_label_split["split"],
        categories=preferred_order,
        ordered=True,
    )

    score_by_label_split = score_by_label_split.sort_values(["split", "label"])

    score_by_label_split.to_csv(
        REPORTS_TABLES_DIR / "temporal_error_score_by_label_split.csv",
        index=False,
    )

    # ==========================================================
    # 6. Falsos positivos y falsos negativos del test
    # ==========================================================

    test_df = labeled_df[labeled_df["split"] == "test"].copy()

    false_positives_test = (
        test_df[test_df["error_type"] == "FP_licit_alerted"]
        .sort_values("score_illicit", ascending=False)
        .head(200)
    )

    false_negatives_test = (
        test_df[test_df["error_type"] == "FN_illicit_missed"]
        .sort_values("score_illicit", ascending=True)
        .head(200)
    )

    case_cols = [
        col for col in [
            "txId",
            "timestep",
            "split",
            "label",
            "score_illicit",
            "predicted_class",
            "risk_level_probability",
            "risk_percentile",
            "priority_level",
            "risk_rank",
            "in_degree",
            "out_degree",
            "total_degree",
            "error_type",
        ]
        if col in labeled_df.columns
    ]

    false_positives_test[case_cols].to_csv(
        REPORTS_TABLES_DIR / "temporal_error_false_positives_test_top.csv",
        index=False,
    )

    false_negatives_test[case_cols].to_csv(
        REPORTS_TABLES_DIR / "temporal_error_false_negatives_test_top.csv",
        index=False,
    )

    # ==========================================================
    # 7. Resumen JSON
    # ==========================================================

    metrics_summary = {
        row["split"]: {
            "total": int(row["total"]),
            "tp": int(row["tp"]),
            "tn": int(row["tn"]),
            "fp": int(row["fp"]),
            "fn": int(row["fn"]),
            "accuracy": float(row["accuracy"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
        }
        for _, row in metrics_by_split.iterrows()
    }

    test_recall_at_k = (
        recall_by_split[recall_by_split["split"] == "test"]
        .set_index("k")["recall_at_k"]
        .to_dict()
    )

    summary = {
        "important_note": (
            "Este análisis separa los resultados por split temporal. "
            "Para la tesis, las métricas principales de generalización deben reportarse sobre test."
        ),
        "split_definition": {
            "train": "timestep <= 34",
            "validation": "35 <= timestep <= 41",
           "test": "timestep >= 42",
        },
        "metrics_by_split": metrics_summary,
        "test_recall_at_k": {
            str(k): float(v)
            for k, v in test_recall_at_k.items()
        },
        "files_generated": {
            "split_label_distribution": "reports/tables/temporal_split_label_distribution.csv",
            "confusion_by_split": "reports/tables/temporal_error_confusion_by_split.csv",
            "metrics_by_split": "reports/tables/temporal_error_metrics_by_split.csv",
            "recall_at_k_by_split": "reports/tables/temporal_error_recall_at_k_by_split.csv",
            "score_by_label_split": "reports/tables/temporal_error_score_by_label_split.csv",
            "false_positives_test": "reports/tables/temporal_error_false_positives_test_top.csv",
            "false_negatives_test": "reports/tables/temporal_error_false_negatives_test_top.csv",
        },
    }

    with open(
        REPORTS_METRICS_DIR / "temporal_error_analysis_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\nNivel 10B completado: validación temporal generada.")
    print("\nMétricas por split:")
    print(metrics_by_split)

    print("\nArchivos generados:")
    print("reports/tables/temporal_split_label_distribution.csv")
    print("reports/tables/temporal_error_confusion_by_split.csv")
    print("reports/tables/temporal_error_metrics_by_split.csv")
    print("reports/tables/temporal_error_recall_at_k_by_split.csv")
    print("reports/tables/temporal_error_score_by_label_split.csv")
    print("reports/tables/temporal_error_false_positives_test_top.csv")
    print("reports/tables/temporal_error_false_negatives_test_top.csv")
    print("reports/metrics/temporal_error_analysis_summary.json")


if __name__ == "__main__":
    main()