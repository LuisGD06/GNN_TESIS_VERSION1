from pathlib import Path
import json
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "elliptic" / "platform_predictions.parquet"
ALERTS_PATH = PROJECT_ROOT / "data" / "processed" / "elliptic" / "platform_alerts_top.parquet"
SUBGRAPH_ALERTS_PATH = PROJECT_ROOT / "data" / "processed" / "elliptic" / "subgraph_alerts.parquet"

REPORTS_TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
REPORTS_METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_predicted_class(value):
    """
    Convierte diferentes formatos de predicted_class a clase binaria:
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


def recall_at_k(df: pd.DataFrame, k: int) -> float:
    labeled = df[df["label"].isin(["licit", "illicit"])].copy()
    positives = (labeled["y_true"] == 1).sum()

    if positives == 0:
        return 0.0

    top_k = labeled.sort_values("score_illicit", ascending=False).head(k)
    found = (top_k["y_true"] == 1).sum()

    return float(found / positives)


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"No se encontró {PREDICTIONS_PATH}")

    predictions_df = pd.read_parquet(PREDICTIONS_PATH)

    required_cols = [
        "txId",
        "timestep",
        "label",
        "score_illicit",
        "predicted_class",
        "priority_level",
        "risk_rank",
    ]

    missing_cols = [col for col in required_cols if col not in predictions_df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas: {missing_cols}")

    df = predictions_df.copy()

    # Solo licit/illicit tienen etiqueta supervisada real.
    labeled_df = df[df["label"].isin(["licit", "illicit"])].copy()

    labeled_df["y_true"] = labeled_df["label"].map(
        {
            "licit": 0,
            "illicit": 1,
        }
    )

    labeled_df["y_pred"] = labeled_df["predicted_class"].apply(normalize_predicted_class)

    # Si predicted_class no vino en 0/1, se reconstruye con threshold 0.5.
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

    # ==========================
    # 1. Matriz de confusión
    # ==========================
    confusion_summary = (
        labeled_df["error_type"]
        .value_counts()
        .rename_axis("error_type")
        .reset_index(name="count")
    )

    total_labeled = len(labeled_df)
    confusion_summary["percentage"] = confusion_summary["count"] / total_labeled

    confusion_summary.to_csv(
        REPORTS_TABLES_DIR / "error_confusion_summary.csv",
        index=False,
    )

    # ==========================
    # 2. Casos de error
    # ==========================
    false_positives = (
        labeled_df[labeled_df["error_type"] == "FP_licit_alerted"]
        .sort_values("score_illicit", ascending=False)
        .head(200)
    )

    false_negatives = (
        labeled_df[labeled_df["error_type"] == "FN_illicit_missed"]
        .sort_values("score_illicit", ascending=True)
        .head(200)
    )

    cols_error_cases = [
        col for col in [
            "txId",
            "timestep",
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

    false_positives[cols_error_cases].to_csv(
        REPORTS_TABLES_DIR / "error_cases_false_positives_top.csv",
        index=False,
    )

    false_negatives[cols_error_cases].to_csv(
        REPORTS_TABLES_DIR / "error_cases_false_negatives_top.csv",
        index=False,
    )

    # ==========================
    # 3. Score por etiqueta
    # ==========================
    score_by_label = (
        df.groupby("label")["score_illicit"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )

    score_by_label.to_csv(
        REPORTS_TABLES_DIR / "error_score_by_label.csv",
        index=False,
    )

    # ==========================
    # 4. Score por timestep
    # ==========================
    score_by_timestep = (
        df.groupby(["timestep", "label"])
        .agg(
            count=("txId", "count"),
            mean_score=("score_illicit", "mean"),
            median_score=("score_illicit", "median"),
            max_score=("score_illicit", "max"),
        )
        .reset_index()
    )

    score_by_timestep.to_csv(
        REPORTS_TABLES_DIR / "error_score_by_timestep.csv",
        index=False,
    )

    # ==========================
    # 5. Prioridad por etiqueta
    # ==========================
    priority_by_label = (
        df.groupby(["priority_level", "label"])
        .size()
        .reset_index(name="count")
    )

    priority_by_label.to_csv(
        REPORTS_TABLES_DIR / "error_priority_by_label.csv",
        index=False,
    )

    # ==========================
    # 6. Recall@K
    # ==========================
    k_values = [50, 100, 200, 500, 1000, 2000, 5000]

    recall_at_k_df = pd.DataFrame(
        {
            "k": k_values,
            "recall_at_k": [recall_at_k(labeled_df, k) for k in k_values],
        }
    )

    recall_at_k_df.to_csv(
        REPORTS_TABLES_DIR / "error_recall_at_k.csv",
        index=False,
    )

    # ==========================
    # 7. Análisis de unknown
    # ==========================
    unknown_df = df[df["label"] == "unknown"].copy()

    unknown_summary = {
        "num_unknown": int(len(unknown_df)),
        "unknown_mean_score": float(unknown_df["score_illicit"].mean()) if len(unknown_df) else 0.0,
        "unknown_median_score": float(unknown_df["score_illicit"].median()) if len(unknown_df) else 0.0,
        "unknown_critical_count": int((unknown_df["priority_level"] == "critical").sum()) if len(unknown_df) else 0,
        "unknown_high_count": int((unknown_df["priority_level"] == "high").sum()) if len(unknown_df) else 0,
    }

    # ==========================
    # 8. Resumen JSON
    # ==========================
    tp = int(((labeled_df["y_true"] == 1) & (labeled_df["y_pred"] == 1)).sum())
    tn = int(((labeled_df["y_true"] == 0) & (labeled_df["y_pred"] == 0)).sum())
    fp = int(((labeled_df["y_true"] == 0) & (labeled_df["y_pred"] == 1)).sum())
    fn = int(((labeled_df["y_true"] == 1) & (labeled_df["y_pred"] == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(labeled_df) if len(labeled_df) else 0.0

    summary = {
        "dataset": {
            "total_predictions": int(len(df)),
            "total_labeled": int(len(labeled_df)),
            "total_unknown": int(len(unknown_df)),
        },
        "confusion_matrix_counts": {
            "tp_illicit_detected": tp,
            "tn_licit_correct": tn,
            "fp_licit_alerted": fp,
            "fn_illicit_missed": fn,
        },
        "metrics_from_platform_predictions": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
        "recall_at_k": {
            str(row["k"]): float(row["recall_at_k"])
            for _, row in recall_at_k_df.iterrows()
        },
        "unknown_risk_summary": unknown_summary,
        "interpretation_notes": {
            "fp": "Falsos positivos: transacciones lícitas priorizadas como riesgosas. En AML pueden representar fricción operativa, pero también casos para revisión preventiva.",
            "fn": "Falsos negativos: transacciones ilícitas no detectadas. En AML son críticos porque representan riesgo no capturado.",
            "unknown": "Los nodos unknown no se usan para evaluación supervisada; sus scores se interpretan como priorización de riesgo.",
        },
    }

    with open(REPORTS_METRICS_DIR / "error_analysis_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\nNivel 10A completado: análisis de errores generado.")
    print(f"Total labeled: {len(labeled_df):,}")
    print(f"TP: {tp:,} | TN: {tn:,} | FP: {fp:,} | FN: {fn:,}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print("\nArchivos generados:")
    print("reports/tables/error_confusion_summary.csv")
    print("reports/tables/error_cases_false_positives_top.csv")
    print("reports/tables/error_cases_false_negatives_top.csv")
    print("reports/tables/error_score_by_label.csv")
    print("reports/tables/error_score_by_timestep.csv")
    print("reports/tables/error_priority_by_label.csv")
    print("reports/tables/error_recall_at_k.csv")
    print("reports/metrics/error_analysis_summary.json")


if __name__ == "__main__":
    main()