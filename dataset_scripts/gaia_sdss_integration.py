from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gaia_sdss.config import GaiaIntegrationConfig
from gaia_sdss.gaia_client import match_rows
from gaia_sdss.local_data import load_local_sdss_dataset
from gaia_sdss.reporting import build_consolidated_dataset, enrich_derived_columns, generate_reports

VALID_CLASSES = ("GALAXY", "QSO", "STAR")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integra dados SDSS locais ja existentes com Gaia DR3 sem baixar SDSS novamente."
    )
    parser.add_argument(
        "--input",
        default="dataset/metadata/spectra_metadata.csv",
        help="Dataset local consolidado ja existente.",
    )
    parser.add_argument(
        "--manifest",
        default="dataset/manifests/download_manifest.csv",
        help="Manifesto local usado para recuperar RA/DEC quando necessario.",
    )
    parser.add_argument(
        "--output",
        default="dataset/gaia_sdss",
        help="Diretorio novo para cache, dataset consolidado, relatorios e graficos.",
    )
    parser.add_argument("--match-radius-arcsec", type=float, default=1.0)
    parser.add_argument("--ambiguity-delta-arcsec", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=VALID_CLASSES,
        default=["STAR"],
        help="Classes SDSS usadas na consulta Gaia. Padrao: STAR.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limita objetos para teste.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconsulta Gaia para os objetos selecionados, preservando cache de outras classes.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Reconsulta apenas objetos selecionados que estao no cache com match_status=ERROR.",
    )
    parser.add_argument("--no-batch", action="store_true", help="Usa consultas individuais em vez de upload TAP.")
    parser.add_argument(
        "--generate-reports-only",
        action="store_true",
        help="Gera relatorios apenas a partir do dataset consolidado existente; nao consulta Gaia.",
    )
    return parser.parse_args()


def resolve_path(path: str, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def filter_by_classes(df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    if "spectralClass" not in df.columns:
        raise ValueError("Dataset local precisa conter a coluna spectralClass para filtrar classes.")

    selected = {value.strip().upper() for value in classes}
    class_series = df["spectralClass"].fillna("").astype(str).str.upper()
    return df[class_series.isin(selected)].copy()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size precisa ser maior ou igual a 1.")

    root = repo_root()
    input_path = resolve_path(args.input, root)
    manifest_path = resolve_path(args.manifest, root)
    output_dir = resolve_path(args.output, root)
    cache_path = output_dir / "cache" / "gaia_matches.csv"
    consolidated_path = output_dir / "gaia_sdss_comparison.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_reports_only:
        if not consolidated_path.exists():
            raise FileNotFoundError(f"Dataset consolidado nao encontrado: {consolidated_path}")
        consolidated = pd.read_csv(consolidated_path, dtype={"specObjID": str, "objID": str})
        if "spectralClass" in consolidated.columns:
            consolidated = filter_by_classes(consolidated, args.classes)
        consolidated = enrich_derived_columns(consolidated)
        consolidated.to_csv(consolidated_path, index=False)
        generate_reports(
            consolidated,
            output_dir,
            sources={
                "mode": "reports_only",
                "consolidated_dataset": str(consolidated_path),
            },
        )
        print(f"Relatorios atualizados em: {output_dir / 'reports'}")
        return 0

    local_df = load_local_sdss_dataset(input_path, manifest_path)
    existing_gaia_columns = local_df.attrs.get("existing_gaia_columns", [])
    total_loaded = len(local_df)
    local_df = filter_by_classes(local_df, args.classes)
    if args.limit is not None:
        local_df = local_df.head(args.limit).copy()

    config = GaiaIntegrationConfig(
        match_radius_arcsec=args.match_radius_arcsec,
        ambiguity_delta_arcsec=args.ambiguity_delta_arcsec,
        batch_size=args.batch_size,
    )

    print(f"Dataset local: {input_path}")
    print(f"Manifesto: {manifest_path}")
    print(f"Classes selecionadas: {', '.join(args.classes)}")
    print(f"Objetos carregados antes do filtro: {total_loaded}")
    print(f"Objetos selecionados para Gaia: {len(local_df)}")
    print(f"Objetos com RA/DEC: {int(local_df['has_radec'].sum())}")
    print(f"Colunas Gaia ja existentes: {existing_gaia_columns or 'nenhuma'}")
    print("Nenhum download SDSS sera executado.")

    gaia_matches = match_rows(
        local_df,
        cache_path=cache_path,
        radius_arcsec=config.match_radius_arcsec,
        ambiguity_delta_arcsec=config.ambiguity_delta_arcsec,
        batch_size=config.batch_size,
        force=args.force,
        retry_errors=args.retry_errors,
        use_batch=not args.no_batch,
    )

    consolidated = build_consolidated_dataset(local_df, gaia_matches)
    consolidated.to_csv(consolidated_path, index=False)

    generate_reports(
        consolidated,
        output_dir,
        sources={
            "local_dataset": str(input_path),
            "manifest": str(manifest_path),
            "gaia_cache": str(cache_path),
            "consolidated_dataset": str(consolidated_path),
            "match_radius_arcsec": config.match_radius_arcsec,
            "ambiguity_delta_arcsec": config.ambiguity_delta_arcsec,
            "classes": args.classes,
        },
    )

    print(f"Cache Gaia: {cache_path}")
    print(f"Dataset consolidado: {consolidated_path}")
    print(f"Relatorio: {output_dir / 'reports' / 'gaia_sdss_report.md'}")
    print(f"Graficos: {output_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
