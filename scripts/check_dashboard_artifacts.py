# Este script sirve para demostrar que tu dashboard tiene todos los artefactos necesarios.
from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = {
    "Predicciones completas": PROJECT_ROOT / "data/processed/elliptic/platform_predictions.parquet",
    "Top alertas": PROJECT_ROOT / "data/processed/elliptic/platform_alerts_top.parquet",
    "Alertas con subgrafos": PROJECT_ROOT / "data/processed/elliptic/subgraph_alerts.parquet",
    "Comparación final de modelos": PROJECT_ROOT / "reports/tables/final_model_comparison.csv",
    "Resumen de métricas": PROJECT_ROOT / "reports/metrics/platform_metrics_summary.json",
    "Metadata de exportación": PROJECT_ROOT / "reports/metrics/platform_export_metadata.json",
}


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def validate_parquet(path: Path) -> tuple[int, int]:
    df = pd.read_parquet(path)
    return df.shape


def validate_csv(path: Path) -> tuple[int, int]:
    df = pd.read_csv(path)
    return df.shape


def validate_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    print("\nValidación de artefactos del dashboard AML\n")

    all_ok = True

    for name, path in REQUIRED_FILES.items():
        print(f"Archivo: {name}")
        print(f"Ruta: {path.relative_to(PROJECT_ROOT)}")

        if not path.exists():
            print("Estado: NO ENCONTRADO\n")
            all_ok = False
            continue

        size = file_size_mb(path)
        print(f"Estado: OK")
        print(f"Tamaño: {size:.2f} MB")

        if path.suffix == ".parquet":
            rows, cols = validate_parquet(path)
            print(f"Dimensión: {rows:,} filas x {cols:,} columnas")

        elif path.suffix == ".csv":
            rows, cols = validate_csv(path)
            print(f"Dimensión: {rows:,} filas x {cols:,} columnas")

        elif path.suffix == ".json":
            content = validate_json(path)
            print(f"Claves JSON: {list(content.keys())[:10]}")

        print("-" * 70)

    if all_ok:
        print("\nValidación completada: todos los artefactos requeridos están disponibles.\n")
    else:
        print("\nValidación incompleta: faltan uno o más artefactos.\n")


if __name__ == "__main__":
    main()