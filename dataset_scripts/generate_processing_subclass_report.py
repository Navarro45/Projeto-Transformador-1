from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from shared.spectral_labels import normalize_spectral_label


def _string_series(frame: pd.DataFrame, column: str, default: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([default] * len(frame), index=frame.index, dtype=str)
    series = frame[column].fillna(default).astype(str).str.strip()
    return series.replace("", default)


def _markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return "Sem dados."
    table = frame.copy()
    if max_rows is not None:
        table = table.head(max_rows)
    table = table.fillna("")
    lines = [
        "| " + " | ".join(map(str, table.columns)) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for _, row in table.iterrows():
        values = [str(row[column]).replace("|", "/") for column in table.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _load_sdss_subclasses(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de metadados SDSS nao encontrado: {path}")

    df = pd.read_csv(path, dtype=str)
    spectral_class = _string_series(df, "spectralClass", "UNKNOWN")
    subclass = _string_series(df, "subclass", "SEM_SUBCLASSE")

    counts = (
        pd.DataFrame({"spectralClass": spectral_class, "subclass": subclass})
        .value_counts(["spectralClass", "subclass"])
        .rename("count")
        .reset_index()
        .sort_values(["spectralClass", "subclass"])
        .reset_index(drop=True)
    )
    class_counts = (
        spectral_class.value_counts()
        .rename_axis("spectralClass")
        .reset_index(name="count")
        .sort_values("spectralClass")
        .reset_index(drop=True)
    )
    return counts, class_counts, len(df)


def _load_star_processing_labels(path: Path, rare_threshold: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    if not path.exists():
        raise FileNotFoundError(f"Indice star_objects nao encontrado: {path}")

    df = pd.read_csv(path, dtype=str)
    raw = _string_series(df, "spectral_subclass_raw", "UNKNOWN")
    labels = pd.DataFrame([normalize_spectral_label(value) for value in raw])

    normalized = pd.concat(
        [
            df.get("star_id", pd.Series([""] * len(df))).rename("star_id"),
            raw.rename("spectral_subclass_raw"),
            labels[["spectral_parent_label", "spectral_leaf_label"]],
        ],
        axis=1,
    )

    leaf_counts = (
        normalized.value_counts(["spectral_parent_label", "spectral_leaf_label"])
        .rename("count")
        .reset_index()
        .sort_values(["spectral_parent_label", "spectral_leaf_label"])
        .reset_index(drop=True)
    )
    leaf_counts["percentage"] = (leaf_counts["count"] / len(normalized) * 100.0).round(4)
    leaf_counts["is_rare"] = leaf_counts["count"] <= rare_threshold

    raw_to_normalized = (
        normalized.value_counts(["spectral_subclass_raw", "spectral_parent_label", "spectral_leaf_label"])
        .rename("count")
        .reset_index()
        .sort_values(["spectral_parent_label", "spectral_leaf_label", "spectral_subclass_raw"])
        .reset_index(drop=True)
    )

    parent_counts = (
        normalized["spectral_parent_label"]
        .value_counts()
        .rename_axis("spectral_parent_label")
        .reset_index(name="count")
        .sort_values("spectral_parent_label")
        .reset_index(drop=True)
    )
    return leaf_counts, raw_to_normalized, parent_counts, len(normalized)


def build_report(
    spectra_metadata: Path,
    star_objects_index: Path,
    output_dir: Path,
    rare_threshold: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    sdss_counts, class_counts, total_spectra = _load_sdss_subclasses(spectra_metadata)
    leaf_counts, raw_to_normalized, parent_counts, total_stars = _load_star_processing_labels(
        star_objects_index,
        rare_threshold,
    )

    sdss_counts_path = output_dir / "processing_sdss_raw_subclasses.csv"
    leaf_counts_path = output_dir / "processing_star_normalized_subclasses.csv"
    raw_mapping_path = output_dir / "processing_star_raw_to_normalized_subclasses.csv"
    parent_counts_path = output_dir / "processing_star_parent_class_counts.csv"
    report_path = output_dir / "processing_subclasses_report.md"
    summary_path = output_dir / "processing_subclasses_summary.json"

    sdss_counts.to_csv(sdss_counts_path, index=False)
    leaf_counts.to_csv(leaf_counts_path, index=False)
    raw_to_normalized.to_csv(raw_mapping_path, index=False)
    parent_counts.to_csv(parent_counts_path, index=False)

    summary = {
        "sources": {
            "spectra_metadata": str(spectra_metadata.resolve()),
            "star_objects_index": str(star_objects_index.resolve()),
        },
        "total_processed_spectra": int(total_spectra),
        "total_star_objects": int(total_stars),
        "sdss_raw_subclass_rows": int(len(sdss_counts)),
        "star_normalized_subclass_rows": int(len(leaf_counts)),
        "rare_threshold": int(rare_threshold),
        "rare_normalized_subclasses": int(leaf_counts["is_rare"].sum()),
        "outputs": {
            "report": str(report_path.resolve()),
            "sdss_raw_subclasses_csv": str(sdss_counts_path.resolve()),
            "star_normalized_subclasses_csv": str(leaf_counts_path.resolve()),
            "star_raw_to_normalized_csv": str(raw_mapping_path.resolve()),
            "star_parent_counts_csv": str(parent_counts_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report_lines = [
        "# Relatorio De Subclasses Usadas No Processamento",
        "",
        "Este relatorio separa dois niveis de rotulo:",
        "",
        "- `SDSS raw`: subclasses originais preservadas no dataset local.",
        "- `STAR normalizado`: subclasses efetivamente usadas como alvo nos modelos supervisionados de estrelas.",
        "",
        "A fonte ativa do projeto e `dataset/`. O diretorio `legacy/data/` e apenas historico e nao deve ser usado como fonte dos experimentos atuais.",
        "",
        "## Fontes",
        "",
        f"- Metadados espectrais: `{spectra_metadata.resolve()}`",
        f"- Indice de objetos STAR: `{star_objects_index.resolve()}`",
        "",
        "## Resumo",
        "",
        f"- Espectros processados no dataset local: {total_spectra}",
        f"- Objetos STAR na camada por objeto: {total_stars}",
        f"- Linhas de classe/subclasse SDSS bruta: {len(sdss_counts)}",
        f"- Subclasses STAR normalizadas usadas no processamento supervisionado: {len(leaf_counts)}",
        f"- Subclasses STAR raras (`count <= {rare_threshold}`): {int(leaf_counts['is_rare'].sum())}",
        "",
        "## Totais Por Classe SDSS",
        "",
        _markdown_table(class_counts),
        "",
        "## Subclasses STAR Normalizadas Usadas No Processamento",
        "",
        _markdown_table(leaf_counts),
        "",
        "## Mapeamento Bruto SDSS -> Subclasse Normalizada STAR",
        "",
        _markdown_table(raw_to_normalized),
        "",
        "## Todas As Subclasses SDSS Brutas Preservadas",
        "",
        _markdown_table(sdss_counts),
        "",
        "## Arquivos CSV Gerados",
        "",
        f"- `{sdss_counts_path}`",
        f"- `{leaf_counts_path}`",
        f"- `{raw_mapping_path}`",
        f"- `{parent_counts_path}`",
        f"- `{summary_path}`",
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera relatorio das subclasses usadas no processamento.")
    parser.add_argument(
        "--spectra-metadata",
        default="dataset/metadata/spectra_metadata.csv",
        help="CSV consolidado com metadados espectrais SDSS.",
    )
    parser.add_argument(
        "--star-objects-index",
        default="dataset/star_objects/star_objects_index.csv",
        help="Indice da camada dataset/star_objects.",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/reports",
        help="Diretorio de saida dos relatorios.",
    )
    parser.add_argument(
        "--rare-threshold",
        type=int,
        default=5,
        help="Limite usado para marcar subclasses STAR raras.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_report(
        Path(args.spectra_metadata),
        Path(args.star_objects_index),
        Path(args.output_dir),
        args.rare_threshold,
    )
    print(f"Relatorio gerado em: {summary['outputs']['report']}")
    print(f"Subclasses STAR normalizadas: {summary['star_normalized_subclass_rows']}")
    print(f"Subclasses STAR raras: {summary['rare_normalized_subclasses']}")


if __name__ == "__main__":
    main()
