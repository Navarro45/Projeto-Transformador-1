from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gaia_sdss.config import GAIA_AMBIGUITY_DELTA_ARCSEC, GAIA_MATCH_RADIUS_ARCSEC
from gaia_sdss.gaia_verified_stars import (
    filter_gaia_verified_candidates,
    load_existing_identifiers,
    match_rows_with_gaia_ap,
    normalize_identifier,
    prepare_sdss_star_candidates,
    read_sdss_candidate_csv,
)
from scripts.download_sdss_dr20_dataset import (
    append_manifest_row,
    process_row,
    write_manifest_header,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Baixa novos objetos STAR do SDSS somente apos confirmar no Gaia DR3 "
            "classe espectral e classe luminosa/evolutiva, sem repetir objetos."
        )
    )
    parser.add_argument(
        "--csv",
        default="dataset/manifests/MyResult_2026824.csv",
        help="CSV SDSS/SkyServer com objID, specObjID, spectralClass, ra e dec.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="Raiz do dataset local onde os novos arquivos SDSS serao salvos.",
    )
    parser.add_argument(
        "--output",
        default="dataset/gaia_verified_sdss",
        help="Diretorio para cache Gaia, candidatos verificados e manifesto do lote.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita o total de candidatos SDSS examinados apos deduplicacao.",
    )
    parser.add_argument(
        "--target-verified",
        type=int,
        default=None,
        help="Quantidade desejada de objetos novos com Gaia confirmado para baixar.",
    )
    parser.add_argument("--match-radius-arcsec", type=float, default=GAIA_MATCH_RADIUS_ARCSEC)
    parser.add_argument("--ambiguity-delta-arcsec", type=float, default=GAIA_AMBIGUITY_DELTA_ARCSEC)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--force-gaia", action="store_true", help="Reconsulta Gaia para os candidatos selecionados.")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Reconsulta apenas candidatos que ficaram com match_status=ERROR no cache Gaia deste fluxo.",
    )
    parser.add_argument(
        "--no-auto-retry-errors",
        action="store_true",
        help="Nao reconsulta automaticamente erros de conexao quando --target-verified esta ativo.",
    )
    parser.add_argument(
        "--gaia-query-retries",
        type=int,
        default=3,
        help="Tentativas por consulta Gaia antes de registrar ERROR.",
    )
    parser.add_argument(
        "--gaia-retry-sleep",
        type=float,
        default=5.0,
        help="Pausa em segundos entre tentativas Gaia.",
    )
    parser.add_argument(
        "--gaia-timeout",
        type=float,
        default=60.0,
        help="Timeout em segundos para cada consulta individual Gaia.",
    )
    parser.add_argument("--no-batch", action="store_true", help="Usa consultas TAP individuais no Gaia.")
    parser.add_argument(
        "--disable-individual-fallback",
        action="store_true",
        help="Nao tenta consultas individuais quando uma consulta Gaia em lote falha.",
    )
    parser.add_argument(
        "--allow-ambiguous",
        action="store_true",
        help="Permite MATCH_AMBIGUOUS; por padrao matches ambiguos nao sao baixados.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Gera CSVs de aprovados/rejeitados, mas nao baixa imagens nem espectros SDSS.",
    )
    parser.add_argument(
        "--keep-rejected",
        action="store_true",
        help="Mantem rejected_candidates.csv com objetos sem Gaia confirmado para auditoria.",
    )
    parser.add_argument("--skip-images", action="store_true", help="Nao baixa JPEGs SDSS mesmo para aprovados.")
    parser.add_argument("--skip-spectra", action="store_true", help="Nao baixa FITS espectrais SDSS.")
    parser.add_argument(
        "--skip-spectrum-processing",
        action="store_true",
        help="Nao gera metadados/plots a partir dos FITS baixados.",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Nao salva PNG dos espectros.")
    parser.add_argument("--save-spectrum-csv", action="store_true", help="Extrai wavelength/flux para CSV.")
    parser.add_argument("--overwrite", action="store_true", help="Rebaixa arquivos SDSS ja existentes.")
    parser.add_argument("--width", type=int, default=256, help="Largura do JPEG SDSS.")
    parser.add_argument("--height", type=int, default=256, help="Altura do JPEG SDSS.")
    parser.add_argument("--scale", type=float, default=0.4, help="Escala do JPEG em arcsec/pixel.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout HTTP para downloads SDSS.")
    parser.add_argument("--retries", type=int, default=3, help="Tentativas por download SDSS.")
    parser.add_argument("--sleep", type=float, default=0.15, help="Pausa entre downloads SDSS.")
    return parser.parse_args()


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def _unlink_if_exists(path: Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()


def _subset_matches_for_candidates(matches: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if matches.empty or candidates.empty or "specObjID" not in matches.columns:
        return pd.DataFrame(columns=matches.columns)

    selected_ids = set(candidates["specObjID"].map(normalize_identifier))
    return matches[matches["specObjID"].map(normalize_identifier).isin(selected_ids)].copy()


def trim_verified_to_target(
    verified: pd.DataFrame,
    rejected: pd.DataFrame,
    target_verified: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if target_verified is None or len(verified) <= target_verified:
        return verified, rejected

    kept = verified.head(target_verified).copy()
    overflow = verified.iloc[target_verified:].copy()
    overflow["rejection_reason"] = "over_target_gaia_confirmed"
    rejected = _concat_frames([rejected, overflow])
    return kept, rejected


def filter_outputs_to_verified(
    candidates: pd.DataFrame,
    gaia_matches: pd.DataFrame,
    verified: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if verified.empty:
        return candidates.iloc[0:0].copy(), gaia_matches.iloc[0:0].copy()

    verified_ids = set(verified["specObjID"].map(normalize_identifier))
    clean_candidates = candidates[
        candidates["specObjID"].map(normalize_identifier).isin(verified_ids)
    ].copy()

    if "specObjID" in gaia_matches.columns:
        clean_matches = gaia_matches[
            gaia_matches["specObjID"].map(normalize_identifier).isin(verified_ids)
        ].copy()
    else:
        clean_matches = gaia_matches.iloc[0:0].copy()

    return clean_candidates, clean_matches


def search_gaia_verified_candidates(
    candidates: pd.DataFrame,
    existing_source_ids: set[str],
    cache_path: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    retry_errors = args.retry_errors or (
        args.target_verified is not None and not args.no_auto_retry_errors
    )

    if args.target_verified is None:
        gaia_matches = match_rows_with_gaia_ap(
            candidates,
            cache_path=cache_path,
            radius_arcsec=args.match_radius_arcsec,
            ambiguity_delta_arcsec=args.ambiguity_delta_arcsec,
            batch_size=args.batch_size,
            force=args.force_gaia,
            retry_errors=retry_errors,
            use_batch=not args.no_batch,
            query_retries=args.gaia_query_retries,
            retry_sleep_seconds=args.gaia_retry_sleep,
            network_timeout=args.gaia_timeout,
            fallback_to_individual=not args.disable_individual_fallback,
        )
        verified, rejected = filter_gaia_verified_candidates(
            candidates,
            gaia_matches,
            existing_gaia_source_ids=existing_source_ids,
            allow_ambiguous=args.allow_ambiguous,
        )
        return verified, rejected, _subset_matches_for_candidates(gaia_matches, candidates), False

    verified_parts: list[pd.DataFrame] = []
    rejected_parts: list[pd.DataFrame] = []
    match_parts: list[pd.DataFrame] = []
    known_source_ids = set(existing_source_ids)
    reached_target = False

    for start in range(0, len(candidates), args.batch_size):
        current_total = sum(len(frame) for frame in verified_parts)
        remaining = args.target_verified - current_total
        if remaining <= 0:
            reached_target = True
            break

        batch = candidates.iloc[start:start + args.batch_size].copy()
        if batch.empty:
            continue

        gaia_matches = match_rows_with_gaia_ap(
            batch,
            cache_path=cache_path,
            radius_arcsec=args.match_radius_arcsec,
            ambiguity_delta_arcsec=args.ambiguity_delta_arcsec,
            batch_size=args.batch_size,
            force=args.force_gaia,
            retry_errors=retry_errors,
            use_batch=not args.no_batch,
            query_retries=args.gaia_query_retries,
            retry_sleep_seconds=args.gaia_retry_sleep,
            network_timeout=args.gaia_timeout,
            fallback_to_individual=not args.disable_individual_fallback,
        )
        batch_matches = _subset_matches_for_candidates(gaia_matches, batch)
        verified_batch, rejected_batch = filter_gaia_verified_candidates(
            batch,
            batch_matches,
            existing_gaia_source_ids=known_source_ids,
            allow_ambiguous=args.allow_ambiguous,
        )
        verified_batch, rejected_batch = trim_verified_to_target(
            verified_batch,
            rejected_batch,
            remaining,
        )

        if not verified_batch.empty and "source_id" in verified_batch.columns:
            known_source_ids.update(
                value
                for value in verified_batch["source_id"].map(normalize_identifier)
                if value
            )

        verified_parts.append(verified_batch)
        rejected_parts.append(rejected_batch)
        match_parts.append(batch_matches)

    verified = _concat_frames(verified_parts)
    rejected = _concat_frames(rejected_parts)
    gaia_matches = _concat_frames(match_parts)
    reached_target = reached_target or len(verified) >= args.target_verified
    return verified, rejected, gaia_matches, reached_target


def download_verified_rows(
    verified: pd.DataFrame,
    dataset_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    manifest_path = output_dir / "download_manifest.csv"
    write_manifest_header(manifest_path)

    results = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "TCC-SDSS-Gaia-verified-star-downloader/1.0"})
        for _, row in verified.iterrows():
            source_index = int(row.get("_source_row", len(results)))
            result = process_row(
                session=session,
                source_index=source_index,
                row=row.to_dict(),
                dataset_dir=dataset_dir,
                args=args,
            )
            append_manifest_row(manifest_path, result)
            merged = {**row.to_dict(), **{f"download_{key}": value for key, value in result.items()}}
            results.append(merged)

            if args.sleep:
                time.sleep(args.sleep)

    return pd.DataFrame(results)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size precisa ser maior ou igual a 1.")
    if args.gaia_query_retries < 1:
        raise ValueError("--gaia-query-retries precisa ser maior ou igual a 1.")
    if args.gaia_retry_sleep < 0:
        raise ValueError("--gaia-retry-sleep precisa ser maior ou igual a 0.")
    if args.gaia_timeout <= 0:
        raise ValueError("--gaia-timeout precisa ser maior que 0.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit precisa ser maior ou igual a 1.")
    if args.target_verified is not None and args.target_verified < 1:
        raise ValueError("--target-verified precisa ser maior ou igual a 1.")

    root = repo_root()
    csv_path = resolve_path(args.csv, root)
    dataset_dir = resolve_path(args.dataset_dir, root)
    output_dir = resolve_path(args.output, root)
    cache_path = output_dir / "cache" / "gaia_ap_matches.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    source = read_sdss_candidate_csv(csv_path)
    existing = load_existing_identifiers(dataset_dir)
    candidates, rejected_before_gaia = prepare_sdss_star_candidates(
        source,
        existing_identifiers=existing,
        limit=args.limit,
    )

    print(f"CSV SDSS: {csv_path}")
    print(f"Dataset local: {dataset_dir}")
    print(f"Saida do fluxo Gaia-verificado: {output_dir}")
    print(f"Linhas no CSV: {len(source)}")
    print(f"Candidatos STAR unicos e ainda ausentes localmente: {len(candidates)}")
    print(f"Rejeitados antes do Gaia: {len(rejected_before_gaia)}")
    if args.target_verified is not None:
        print(f"Meta de objetos com Gaia confirmado: {args.target_verified}")

    verified, rejected_after_gaia, gaia_matches, reached_target = search_gaia_verified_candidates(
        candidates,
        existing_source_ids=existing.get("source_id", set()),
        cache_path=cache_path,
        args=args,
    )
    verified, rejected_after_gaia = trim_verified_to_target(
        verified,
        rejected_after_gaia,
        args.target_verified,
    )

    rejected = pd.concat(
        [frame for frame in (rejected_before_gaia, rejected_after_gaia) if not frame.empty],
        ignore_index=True,
    ) if not rejected_before_gaia.empty or not rejected_after_gaia.empty else pd.DataFrame()

    if args.keep_rejected:
        output_candidates = candidates
        output_matches = gaia_matches
    else:
        output_candidates, output_matches = filter_outputs_to_verified(candidates, gaia_matches, verified)

    _write_csv(output_candidates, output_dir / "candidate_stars_after_dedup.csv")
    _write_csv(output_matches, output_dir / "gaia_ap_matches.csv")
    _write_csv(verified, output_dir / "verified_gaia_sdss_stars.csv")
    if args.keep_rejected:
        _write_csv(rejected, output_dir / "rejected_candidates.csv")
    else:
        _unlink_if_exists(output_dir / "rejected_candidates.csv")

    if args.dry_run:
        download_results = pd.DataFrame()
        print("Dry-run ativo: nenhum download SDSS foi executado.")
    else:
        download_results = download_verified_rows(verified, dataset_dir, output_dir, args)
        _write_csv(download_results, output_dir / "download_results.csv")

    summary = {
        "source_csv": str(csv_path),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "gaia_cache": str(cache_path),
        "source_rows": int(len(source)),
        "candidates_after_dedup": int(len(candidates)),
        "rejected_before_gaia": int(len(rejected_before_gaia)),
        "gaia_matches_cached_or_queried": int(len(gaia_matches)),
        "verified_for_download": int(len(verified)),
        "target_verified": args.target_verified,
        "target_reached": bool(reached_target),
        "rejected_after_gaia": int(len(rejected_after_gaia)),
        "rejected_rows_deleted_from_clean_outputs": int(0 if args.keep_rejected else len(rejected)),
        "keep_rejected": bool(args.keep_rejected),
        "auto_retry_errors": bool(args.target_verified is not None and not args.no_auto_retry_errors),
        "gaia_query_retries": int(args.gaia_query_retries),
        "gaia_retry_sleep": float(args.gaia_retry_sleep),
        "gaia_timeout": float(args.gaia_timeout),
        "gaia_individual_fallback": bool(not args.disable_individual_fallback),
        "downloaded_rows": int(len(download_results)),
        "dry_run": bool(args.dry_run),
        "allow_ambiguous": bool(args.allow_ambiguous),
        "match_radius_arcsec": float(args.match_radius_arcsec),
        "ambiguity_delta_arcsec": float(args.ambiguity_delta_arcsec),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Cache Gaia AP: {cache_path}")
    print(f"Aprovados para download: {output_dir / 'verified_gaia_sdss_stars.csv'}")
    if args.keep_rejected:
        print(f"Rejeitados auditaveis: {output_dir / 'rejected_candidates.csv'}")
    else:
        print("Objetos sem Gaia confirmado foram removidos dos CSVs finais limpos.")
    if args.target_verified is not None and not reached_target:
        print(
            "Aviso: a meta de Gaia confirmado nao foi atingida com os candidatos disponiveis "
            f"({len(verified)}/{args.target_verified})."
        )
    if not args.dry_run:
        print(f"Manifesto do lote: {output_dir / 'download_manifest.csv'}")
        print(f"Resultados de download: {output_dir / 'download_results.csv'}")
    print(f"Resumo: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        raise SystemExit(2) from None
