from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("dataset/gaia_verified_sdss/verified_gaia_sdss_stars.csv")
DEFAULT_OUTPUT_DIR = Path("dataset/gaia_verified_sdss")

SPECTRAL_ORDER = {"O": 0, "B": 1, "A": 2, "F": 3, "G": 4, "K": 5, "M": 6, "CSTAR": 7, "UNKNOWN": 8}
LUMINOSITY_ORDER = {
    "MAIN_SEQUENCE": 0,
    "SUBGIANT": 1,
    "RED_GIANT_BRANCH": 2,
    "RGB_BUMP": 3,
    "RGB_TIP": 4,
    "UNKNOWN": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume as classes espectrais e luminosas Gaia dos objetos SDSS verificados."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV final de objetos verificados pelo Gaia.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Diretorio onde os resumos serao salvos.")
    parser.add_argument(
        "--update-summary",
        action="store_true",
        help="Atualiza tambem o summary.json do diretorio de saida com a secao gaia_class_summary.",
    )
    return parser.parse_args()


def _clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().replace("", "UNKNOWN")


def _ordered_counts(frame: pd.DataFrame, column: str, order: dict[str, int], total: int) -> pd.DataFrame:
    counts = _clean_text(frame[column]).value_counts().rename_axis(column).reset_index(name="count")
    counts["percent"] = (counts["count"] / total * 100).round(4) if total else 0.0
    counts["_order"] = counts[column].map(order).fillna(len(order)).astype(int)
    return counts.sort_values(["_order", column]).drop(columns="_order").reset_index(drop=True)


def _combined_counts(frame: pd.DataFrame, total: int) -> pd.DataFrame:
    working = pd.DataFrame(
        {
            "gaia_spectral_class": _clean_text(frame["gaia_spectral_class"]),
            "gaia_luminosity_class": _clean_text(frame["gaia_luminosity_class"]),
        }
    )
    counts = (
        working.groupby(["gaia_spectral_class", "gaia_luminosity_class"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    counts["percent"] = (counts["count"] / total * 100).round(4) if total else 0.0
    counts["_spectral_order"] = counts["gaia_spectral_class"].map(SPECTRAL_ORDER).fillna(len(SPECTRAL_ORDER)).astype(int)
    counts["_luminosity_order"] = counts["gaia_luminosity_class"].map(LUMINOSITY_ORDER).fillna(len(LUMINOSITY_ORDER)).astype(int)
    return (
        counts.sort_values(["_spectral_order", "_luminosity_order", "gaia_spectral_class", "gaia_luminosity_class"])
        .drop(columns=["_spectral_order", "_luminosity_order"])
        .reset_index(drop=True)
    )


def _non_empty_unique(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    values = frame[column].fillna("").astype(str).str.strip()
    return int(values[values != ""].nunique())


def _status_counts(frame: pd.DataFrame) -> dict[str, int]:
    if "gaia_required_status" not in frame.columns:
        return {}
    counts = _clean_text(frame["gaia_required_status"]).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"CSV nao encontrado: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input, dtype=str, low_memory=False)
    required_columns = {"gaia_spectral_class", "gaia_luminosity_class"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"CSV precisa conter as colunas: {', '.join(sorted(missing))}")

    total = int(len(frame))
    spectral_counts = _ordered_counts(frame, "gaia_spectral_class", SPECTRAL_ORDER, total)
    luminosity_counts = _ordered_counts(frame, "gaia_luminosity_class", LUMINOSITY_ORDER, total)
    combined_counts = _combined_counts(frame, total)

    spectral_path = args.output_dir / "gaia_spectral_class_counts.csv"
    luminosity_path = args.output_dir / "gaia_luminosity_class_counts.csv"
    combined_path = args.output_dir / "gaia_spectral_luminosity_class_counts.csv"
    summary_path = args.output_dir / "gaia_class_summary.json"

    spectral_counts.to_csv(spectral_path, index=False)
    luminosity_counts.to_csv(luminosity_path, index=False)
    combined_counts.to_csv(combined_path, index=False)

    status_counts = _status_counts(frame)
    class_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(args.input.resolve()),
        "total_rows": total,
        "unique_objID": _non_empty_unique(frame, "objID"),
        "unique_specObjID": _non_empty_unique(frame, "specObjID"),
        "unique_gaia_source_id": _non_empty_unique(frame, "source_id"),
        "duplicate_objID": total - _non_empty_unique(frame, "objID"),
        "duplicate_specObjID": total - _non_empty_unique(frame, "specObjID"),
        "duplicate_gaia_source_id": total - _non_empty_unique(frame, "source_id"),
        "gaia_required_status_counts": status_counts,
        "spectral_class_counts": {
            row["gaia_spectral_class"]: int(row["count"]) for _, row in spectral_counts.iterrows()
        },
        "luminosity_class_counts": {
            row["gaia_luminosity_class"]: int(row["count"]) for _, row in luminosity_counts.iterrows()
        },
        "outputs": {
            "spectral_class_counts_csv": str(spectral_path.resolve()),
            "luminosity_class_counts_csv": str(luminosity_path.resolve()),
            "spectral_luminosity_class_counts_csv": str(combined_path.resolve()),
            "class_summary_json": str(summary_path.resolve()),
        },
    }
    _write_json(summary_path, class_summary)

    if args.update_summary:
        run_summary_path = args.output_dir / "summary.json"
        if run_summary_path.exists():
            run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
        else:
            run_summary = {}
        run_summary["gaia_class_summary"] = class_summary
        _write_json(run_summary_path, run_summary)

    print(f"Total verificado Gaia: {total}")
    print(f"Resumo JSON: {summary_path}")
    print(f"Classes espectrais: {spectral_path}")
    print(f"Classes luminosas/evolutivas: {luminosity_path}")
    print(f"Combinado espectral+luminoso: {combined_path}")


if __name__ == "__main__":
    main()
