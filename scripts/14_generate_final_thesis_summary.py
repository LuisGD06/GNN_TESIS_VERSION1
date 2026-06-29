from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS_METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
REPORTS_TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "elliptic"

OUTPUT_JSON = REPORTS_METRICS_DIR / "final_thesis_analysis_summary.json"
OUTPUT_TABLE = REPORTS_TABLES_DIR / "final_thesis_key_results.csv"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def add_row(rows: list[dict], section: str, metric: str, value, interpretation: str = "") -> None:
    rows.append(
        {
            "section": section,
            "metric": metric,
            "value": value,
            "interpretation": interpretation,
        }
    )


def main() -> None:
    REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    platform_metrics = load_json(REPORTS_METRICS_DIR / "platform_metrics_summary.json")
    temporal_summary = load_json(REPORTS_METRICS_DIR / "temporal_error_analysis_summary.json")
    error_summary = load_json(REPORTS_METRICS_DIR / "error_analysis_summary.json")
    explainability_summary = load_json(REPORTS_METRICS_DIR / "xgb_explainability_summary.json")
    integrity_summary = load_json(REPORTS_METRICS_DIR / "platform_integrity_report.json")

    final_model_comparison_path = REPORTS_TABLES_DIR / "final_model_comparison.csv"
    predictions_path = PROCESSED_DIR / "platform_predictions.parquet"

    rows = []

    # ==========================================================
    # 1. Modelo principal
    # ==========================================================

    primary_model = platform_metrics.get("primary_model", {})

    add_row(
        rows,
        "Modelo principal",
        "Modelo seleccionado",
        primary_model.get("name", "No disponible"),
        "Modelo usado como motor predictivo principal del prototipo.",
    )

    add_row(
        rows,
        "Modelo principal",
        "PR-AUC test",
        round(safe_float(primary_model.get("test_pr_auc")), 6),
        "Métrica principal usada por el desbalance de clases.",
    )

    add_row(
        rows,
        "Modelo principal",
        "ROC-AUC test",
        round(safe_float(primary_model.get("test_roc_auc")), 6),
        "Capacidad general de separación entre clases.",
    )

    add_row(
        rows,
        "Modelo principal",
        "Precision test",
        round(safe_float(primary_model.get("test_precision")), 6),
        "Proporción de alertas positivas que realmente fueron ilícitas.",
    )

    add_row(
        rows,
        "Modelo principal",
        "Recall test",
        round(safe_float(primary_model.get("test_recall")), 6),
        "Proporción de ilícitas detectadas por el modelo.",
    )

    add_row(
        rows,
        "Modelo principal",
        "F1 test",
        round(safe_float(primary_model.get("test_f1")), 6),
        "Balance entre precision y recall.",
    )

    # ==========================================================
    # 2. Comparación de modelos
    # ==========================================================

    model_ranking = []

    if final_model_comparison_path.exists():
        model_comparison_df = pd.read_csv(final_model_comparison_path)
        model_comparison_df = model_comparison_df.sort_values("pr_auc", ascending=False)
        top_models_df = model_comparison_df.head(5).copy()

        for rank, row in enumerate(top_models_df.itertuples(index=False), start=1):
            model_ranking.append(
                {
                    "rank": rank,
                    "model": row.model,
                    "family": row.family,
                    "pr_auc": safe_float(row.pr_auc),
                    "roc_auc": safe_float(row.roc_auc),
                    "precision": safe_float(row.precision),
                    "recall": safe_float(row.recall),
                    "f1": safe_float(row.f1),
                }
            )

        best_model = top_models_df.iloc[0]

        add_row(
            rows,
            "Comparación de modelos",
            "Mejor modelo por PR-AUC",
            best_model["model"],
            "El ranking confirma el modelo principal seleccionado.",
        )

        add_row(
            rows,
            "Comparación de modelos",
            "Cantidad de modelos comparados",
            len(model_comparison_df),
            "Incluye baselines, GNN y modelos con subgrafos.",
        )

    # ==========================================================
    # 3. Dataset y predicciones de plataforma
    # ==========================================================

    platform_counts = {}

    if predictions_path.exists():
        predictions_df = pd.read_parquet(predictions_path)

        label_counts = predictions_df["label"].value_counts().to_dict()
        priority_counts = predictions_df["priority_level"].value_counts().to_dict()

        platform_counts = {
            "total_predictions": int(len(predictions_df)),
            "label_counts": {str(k): int(v) for k, v in label_counts.items()},
            "priority_counts": {str(k): int(v) for k, v in priority_counts.items()},
            "score_min": float(predictions_df["score_illicit"].min()),
            "score_max": float(predictions_df["score_illicit"].max()),
            "score_mean": float(predictions_df["score_illicit"].mean()),
        }

        add_row(
            rows,
            "Predicciones de plataforma",
            "Total de predicciones",
            platform_counts["total_predictions"],
            "Cantidad total de nodos con score generado.",
        )

        add_row(
            rows,
            "Predicciones de plataforma",
            "Nodos illicit",
            platform_counts["label_counts"].get("illicit", 0),
            "Transacciones con etiqueta ilícita conocida.",
        )

        add_row(
            rows,
            "Predicciones de plataforma",
            "Nodos licit",
            platform_counts["label_counts"].get("licit", 0),
            "Transacciones con etiqueta lícita conocida.",
        )

        add_row(
            rows,
            "Predicciones de plataforma",
            "Nodos unknown",
            platform_counts["label_counts"].get("unknown", 0),
            "Transacciones sin etiqueta confirmada.",
        )

        add_row(
            rows,
            "Predicciones de plataforma",
            "Alertas critical",
            platform_counts["priority_counts"].get("critical", 0),
            "Transacciones ubicadas en el percentil más alto de riesgo.",
        )

        add_row(
            rows,
            "Predicciones de plataforma",
            "Alertas high",
            platform_counts["priority_counts"].get("high", 0),
            "Transacciones de prioridad alta.",
        )

    # ==========================================================
    # 4. Validación temporal
    # ==========================================================

    temporal_metrics = temporal_summary.get("metrics_by_split", {})
    test_metrics = temporal_metrics.get("test", {})

    add_row(
        rows,
        "Validación temporal",
        "Total test",
        safe_int(test_metrics.get("total")),
        "Cantidad de nodos etiquetados evaluados en test.",
    )

    add_row(
        rows,
        "Validación temporal",
        "Precision test temporal",
        round(safe_float(test_metrics.get("precision")), 6),
        "Precision calculada respetando el split temporal original.",
    )

    add_row(
        rows,
        "Validación temporal",
        "Recall test temporal",
        round(safe_float(test_metrics.get("recall")), 6),
        "Recall calculado sobre datos futuros del split test.",
    )

    add_row(
        rows,
        "Validación temporal",
        "F1 test temporal",
        round(safe_float(test_metrics.get("f1")), 6),
        "F1 calculado sobre el split test.",
    )

    test_recall_at_k = temporal_summary.get("test_recall_at_k", {})

    for k in ["50", "100", "500", "1000"]:
        if k in test_recall_at_k:
            add_row(
                rows,
                "Validación temporal",
                f"Recall@{k} test",
                round(safe_float(test_recall_at_k[k]), 6),
                "Proporción de ilícitos recuperados en las primeras K alertas.",
            )

    # ==========================================================
    # 5. Análisis de errores
    # ==========================================================

    confusion_counts = error_summary.get("confusion_matrix_counts", {})

    add_row(
        rows,
        "Análisis de errores",
        "TP illicit detected",
        safe_int(confusion_counts.get("tp_illicit_detected")),
        "Ilícitas detectadas correctamente en análisis global supervisado.",
    )

    add_row(
        rows,
        "Análisis de errores",
        "FP licit alerted",
        safe_int(confusion_counts.get("fp_licit_alerted")),
        "Lícitas marcadas como riesgo.",
    )

    add_row(
        rows,
        "Análisis de errores",
        "FN illicit missed",
        safe_int(confusion_counts.get("fn_illicit_missed")),
        "Ilícitas no detectadas.",
    )

    # ==========================================================
    # 6. Explicabilidad
    # ==========================================================

    top_shap = explainability_summary.get("top_10_shap_features", [])
    top_model_importance = explainability_summary.get("top_10_model_importance_features", [])

    add_row(
        rows,
        "Explicabilidad XGBoost",
        "Muestra SHAP",
        explainability_summary.get("shap_sample", {}).get("sample_size", 0),
        "Cantidad de registros del split test usados para SHAP.",
    )

    add_row(
        rows,
        "Explicabilidad XGBoost",
        "Top 10 SHAP",
        ", ".join(top_shap),
        "Variables con mayor contribución promedio al score.",
    )

    add_row(
        rows,
        "Explicabilidad XGBoost",
        "Top 10 importancia interna",
        ", ".join(top_model_importance),
        "Variables más importantes según la importancia interna del modelo.",
    )

    # ==========================================================
    # 7. Integridad técnica
    # ==========================================================

    add_row(
        rows,
        "Integridad técnica",
        "Estado general",
        integrity_summary.get("overall_status", "No disponible"),
        "Resultado del script de validación técnica del prototipo.",
    )

    # ==========================================================
    # 8. Salidas finales
    # ==========================================================

    key_results_df = pd.DataFrame(rows)
    key_results_df.to_csv(OUTPUT_TABLE, index=False)

    final_summary = {
        "project_status": "codigo_y_analisis_completados",
        "primary_model": primary_model,
        "model_ranking_top_5": model_ranking,
        "platform_counts": platform_counts,
        "temporal_test_metrics": test_metrics,
        "test_recall_at_k": test_recall_at_k,
        "error_analysis_counts": confusion_counts,
        "explainability": {
            "top_10_shap_features": top_shap,
            "top_10_model_importance_features": top_model_importance,
            "shap_sample": explainability_summary.get("shap_sample", {}),
        },
        "technical_integrity": {
            "overall_status": integrity_summary.get("overall_status", "No disponible"),
        },
        "main_interpretation": (
            "El prototipo queda técnicamente preparado como plataforma analítica AML. "
            "XGBoost funciona como motor predictivo principal, mientras que grafos, subgrafos, "
            "validación temporal, análisis de errores, vecindario transaccional y SHAP complementan "
            "la interpretación del riesgo."
        ),
        "main_limitations": [
            "Los nodos unknown no tienen etiqueta confirmada.",
            "El módulo de inferencia completa requiere las 168 features del entrenamiento.",
            "La plataforma no reconstruye automáticamente nuevas features blockchain externas.",
            "El score debe interpretarse como apoyo a priorización, no como decisión automática definitiva.",
            "Las features originales del dataset Elliptic están anonimizadas, por lo que su interpretación semántica directa es limitada.",
        ],
        "outputs": {
            "final_thesis_key_results": str(OUTPUT_TABLE),
            "final_thesis_analysis_summary": str(OUTPUT_JSON),
        },
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(final_summary, file, indent=2, ensure_ascii=False)

    print("\nResumen final analítico ")
    print(f"Tabla: {OUTPUT_TABLE}")
    print(f"JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()