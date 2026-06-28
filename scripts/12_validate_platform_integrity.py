from pathlib import Path
import json
import traceback

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_PATH = PROJECT_ROOT / "reports" / "metrics" / "platform_integrity_report.json"


REQUIRED_FILES = {
    "platform_predictions": PROJECT_ROOT / "data" / "processed" / "elliptic" / "platform_predictions.parquet",
    "platform_alerts_top": PROJECT_ROOT / "data" / "processed" / "elliptic" / "platform_alerts_top.parquet",
    "subgraph_alerts": PROJECT_ROOT / "data" / "processed" / "elliptic" / "subgraph_alerts.parquet",
    "platform_edges": PROJECT_ROOT / "data" / "processed" / "elliptic" / "platform_edges.parquet",
    "final_model_comparison": PROJECT_ROOT / "reports" / "tables" / "final_model_comparison.csv",
    "platform_metrics_summary": PROJECT_ROOT / "reports" / "metrics" / "platform_metrics_summary.json",
    "error_analysis_summary": PROJECT_ROOT / "reports" / "metrics" / "error_analysis_summary.json",
    "temporal_error_analysis_summary": PROJECT_ROOT / "reports" / "metrics" / "temporal_error_analysis_summary.json",
    "inference_schema_xgboost": PROJECT_ROOT / "reports" / "metrics" / "inference_schema_xgboost.json",
    "inference_template_xgboost": PROJECT_ROOT / "data" / "processed" / "elliptic" / "inference_template_xgboost.csv",
    "baseline_xgboost_model": PROJECT_ROOT / "models" / "baseline_xgboost.pkl",
}


REQUIRED_PAGES = {
    "dashboard_aml": PROJECT_ROOT / "app" / "pages" / "1_dashboard_aml.py",
    "alertas_priorizadas": PROJECT_ROOT / "app" / "pages" / "2_alertas_priorizadas.py",
    "comparacion_modelos": PROJECT_ROOT / "app" / "pages" / "3_comparacion_modelos.py",
    "analisis_subgrafos": PROJECT_ROOT / "app" / "pages" / "4_analisis_subgrafos.py",
    "detalle_transaccion": PROJECT_ROOT / "app" / "pages" / "5_detalle_transaccion.py",
    "analisis_errores": PROJECT_ROOT / "app" / "pages" / "6_analisis_errores.py",
    "validacion_temporal": PROJECT_ROOT / "app" / "pages" / "7_validacion_temporal.py",
    "carga_scoring_aml": PROJECT_ROOT / "app" / "pages" / "8_carga_scoring_aml.py",
    "vecindario_transaccional": PROJECT_ROOT / "app" / "pages" / "9_vecindario_transaccional.py",
    "inferencia_xgboost": PROJECT_ROOT / "app" / "pages" / "10_inferencia_xgboost.py",
}


PLATFORM_PREDICTION_REQUIRED_COLUMNS = [
    "txId",
    "timestep",
    "label",
    "in_degree",
    "out_degree",
    "total_degree",
    "score_illicit",
    "predicted_class",
    "risk_level_probability",
    "risk_percentile",
    "priority_level",
    "risk_rank",
]


def check_files() -> dict:
    results = {}

    for name, path in REQUIRED_FILES.items():
        results[name] = {
            "path": str(path),
            "exists": path.exists(),
            "size_mb": round(path.stat().st_size / 1024 / 1024, 4) if path.exists() else 0,
        }

    return results


def check_pages() -> dict:
    results = {}

    for name, path in REQUIRED_PAGES.items():
        results[name] = {
            "path": str(path),
            "exists": path.exists(),
        }

    return results


def check_platform_predictions() -> dict:
    path = REQUIRED_FILES["platform_predictions"]

    if not path.exists():
        return {
            "status": "failed",
            "error": "platform_predictions.parquet no existe",
        }

    df = pd.read_parquet(path)

    missing_columns = [
        col for col in PLATFORM_PREDICTION_REQUIRED_COLUMNS
        if col not in df.columns
    ]

    label_counts = (
        df["label"]
        .value_counts(dropna=False)
        .to_dict()
        if "label" in df.columns
        else {}
    )

    priority_counts = (
        df["priority_level"]
        .value_counts(dropna=False)
        .to_dict()
        if "priority_level" in df.columns
        else {}
    )

    return {
        "status": "passed" if not missing_columns else "failed",
        "shape": list(df.shape),
        "missing_columns": missing_columns,
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "priority_counts": {str(k): int(v) for k, v in priority_counts.items()},
        "score_min": float(df["score_illicit"].min()) if "score_illicit" in df.columns else None,
        "score_max": float(df["score_illicit"].max()) if "score_illicit" in df.columns else None,
    }


def check_inference_schema_and_template() -> dict:
    schema_path = REQUIRED_FILES["inference_schema_xgboost"]
    template_path = REQUIRED_FILES["inference_template_xgboost"]

    if not schema_path.exists():
        return {
            "status": "failed",
            "error": "No existe inference_schema_xgboost.json",
        }

    if not template_path.exists():
        return {
            "status": "failed",
            "error": "No existe inference_template_xgboost.csv",
        }

    with open(schema_path, "r", encoding="utf-8") as file:
        schema = json.load(file)

    required_features = schema.get("required_features", [])
    template_df = pd.read_csv(template_path)

    missing_template_features = [
        col for col in required_features
        if col not in template_df.columns
    ]

    return {
        "status": "passed" if not missing_template_features else "failed",
        "num_required_features": len(required_features),
        "template_shape": list(template_df.shape),
        "missing_template_features": missing_template_features,
        "has_txId": "txId" in template_df.columns,
    }


def check_xgboost_inference_smoke_test() -> dict:
    model_path = REQUIRED_FILES["baseline_xgboost_model"]
    schema_path = REQUIRED_FILES["inference_schema_xgboost"]
    template_path = REQUIRED_FILES["inference_template_xgboost"]

    if not model_path.exists():
        return {
            "status": "skipped",
            "reason": "No existe el modelo local baseline_xgboost.pkl. Esto puede ser normal en despliegue cloud.",
        }

    if not schema_path.exists() or not template_path.exists():
        return {
            "status": "failed",
            "error": "Falta schema o template de inferencia",
        }

    try:
        import joblib

        model = joblib.load(model_path)

        with open(schema_path, "r", encoding="utf-8") as file:
            schema = json.load(file)

        required_features = schema.get("required_features", [])
        template_df = pd.read_csv(template_path)

        X = template_df[required_features].astype(float)

        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[:, 1]
        else:
            scores = model.predict(X)

        return {
            "status": "passed",
            "num_predictions": int(len(scores)),
            "score_min": float(min(scores)),
            "score_max": float(max(scores)),
        }

    except Exception as error:
        return {
            "status": "failed",
            "error": str(error),
            "traceback": traceback.format_exc(),
        }


def summarize_status(report: dict) -> str:
    failed_checks = []

    for section_name, section_value in report.items():
        if isinstance(section_value, dict):
            if section_value.get("status") == "failed":
                failed_checks.append(section_name)

    file_missing = [
        name for name, item in report.get("files", {}).items()
        if not item.get("exists", False)
    ]

    page_missing = [
        name for name, item in report.get("pages", {}).items()
        if not item.get("exists", False)
    ]

    if failed_checks or file_missing or page_missing:
        return "warning"

    return "passed"


def main() -> None:
    report = {
        "project_root": str(PROJECT_ROOT),
        "files": check_files(),
        "pages": check_pages(),
        "platform_predictions_check": check_platform_predictions(),
        "inference_schema_template_check": check_inference_schema_and_template(),
        "xgboost_inference_smoke_test": check_xgboost_inference_smoke_test(),
    }

    report["overall_status"] = summarize_status(report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nValidación técnica del prototipo completada.")
    print(f"Estado general: {report['overall_status']}")
    print(f"Reporte generado: {REPORT_PATH}")

    print("\nChecks principales:")
    print(f"- Platform predictions: {report['platform_predictions_check'].get('status')}")
    print(f"- Schema/template inferencia: {report['inference_schema_template_check'].get('status')}")
    print(f"- Smoke test XGBoost: {report['xgboost_inference_smoke_test'].get('status')}")

    missing_files = [
        name for name, item in report["files"].items()
        if not item["exists"]
    ]

    if missing_files:
        print("\nArchivos faltantes:")
        for item in missing_files:
            print(f"- {item}")

    missing_pages = [
        name for name, item in report["pages"].items()
        if not item["exists"]
    ]

    if missing_pages:
        print("\nPáginas faltantes:")
        for item in missing_pages:
            print(f"- {item}")


if __name__ == "__main__":
    main()