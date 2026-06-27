from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERIM_EDGES_PATH = PROJECT_ROOT / "data" / "interim" / "elliptic" / "edges_interim.parquet"
RAW_EDGES_PATH = PROJECT_ROOT / "data" / "raw" / "elliptic" / "elliptic_txs_edgelist.csv"

PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "elliptic" / "platform_predictions.parquet"

OUTPUT_EDGES_PATH = PROJECT_ROOT / "data" / "processed" / "elliptic" / "platform_edges.parquet"


def load_edges() -> pd.DataFrame:
    if INTERIM_EDGES_PATH.exists():
        print(f"Leyendo aristas desde: {INTERIM_EDGES_PATH}")
        edges_df = pd.read_parquet(INTERIM_EDGES_PATH)
    elif RAW_EDGES_PATH.exists():
        print(f"Leyendo aristas desde: {RAW_EDGES_PATH}")
        edges_df = pd.read_csv(RAW_EDGES_PATH)
    else:
        raise FileNotFoundError(
            "No se encontró archivo de aristas. "
            "Se esperaba data/interim/elliptic/edges_interim.parquet "
            "o data/raw/elliptic/elliptic_txs_edgelist.csv"
        )

    return edges_df


def normalize_edges(edges_df: pd.DataFrame) -> pd.DataFrame:
    cols = list(edges_df.columns)

    source_candidates = ["source", "src", "txId1", "txid1", "input", "from"]
    target_candidates = ["target", "dst", "txId2", "txid2", "output", "to"]

    source_col = next((col for col in source_candidates if col in cols), None)
    target_col = next((col for col in target_candidates if col in cols), None)

    if source_col is None or target_col is None:
        if len(cols) < 2:
            raise ValueError("El archivo de aristas debe tener al menos dos columnas.")
        source_col, target_col = cols[0], cols[1]

    normalized = edges_df[[source_col, target_col]].copy()
    normalized.columns = ["source", "target"]

    normalized["source"] = normalized["source"].astype(str).str.strip()
    normalized["target"] = normalized["target"].astype(str).str.strip()

    normalized = normalized.dropna()
    normalized = normalized[
        (normalized["source"] != "")
        & (normalized["target"] != "")
    ]

    normalized = normalized.drop_duplicates()

    return normalized


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"No se encontró {PREDICTIONS_PATH}")

    predictions_df = pd.read_parquet(PREDICTIONS_PATH)
    valid_txids = set(predictions_df["txId"].astype(str).str.strip())

    edges_df = load_edges()
    edges_df = normalize_edges(edges_df)

    print(f"Aristas originales normalizadas: {len(edges_df):,}")

    # Conserva solo aristas cuyos nodos existen en platform_predictions.
    edges_df = edges_df[
        edges_df["source"].isin(valid_txids)
        & edges_df["target"].isin(valid_txids)
    ].copy()

    print(f"Aristas compatibles con plataforma: {len(edges_df):,}")

    OUTPUT_EDGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    edges_df.to_parquet(OUTPUT_EDGES_PATH, index=False)

    print(f"Archivo generado: {OUTPUT_EDGES_PATH}")
    print(f"Tamaño final: {OUTPUT_EDGES_PATH.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()