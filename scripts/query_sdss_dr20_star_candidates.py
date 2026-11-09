from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gaia_sdss.gaia_verified_stars import load_existing_identifiers, normalize_identifier, prepare_sdss_star_candidates


SKYSERVER_SQL_URL = "https://skyserver.sdss.org/dr20/SkyServerWS/SearchTools/SqlSearch"

OUTPUT_COLUMNS = [
    "objID",
    "ra",
    "dec",
    "type",
    "specObjID",
    "spectralClass",
    "redshift",
    "redshiftError",
    "zWarning",
    "plate",
    "mjd",
    "fiberID",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consulta o SDSS DR20 no SkyServer e gera CSV de candidatos STAR para validacao Gaia."
    )
    parser.add_argument(
        "--output",
        default="dataset/gaia_sdss/cache/new_gaia_candidates.csv",
        help="CSV de saida usado depois por download_gaia_verified_sdss_stars.py.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="Raiz do dataset local usada para evitar objetos ja existentes.",
    )
    parser.add_argument(
        "--target-candidates",
        type=int,
        default=50000,
        help="Quantidade desejada de candidatos SDSS novos no CSV de saida.",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=5000,
        help="Quantidade solicitada ao SkyServer por consulta.",
    )
    parser.add_argument(
        "--min-specobjid",
        default=None,
        help="Comeca a busca apos este specObjID. Por padrao retoma do maior specObjID ja salvo.",
    )
    parser.add_argument(
        "--include-zwarning",
        action="store_true",
        help="Inclui espectros com zWarning diferente de 0.",
    )
    parser.add_argument(
        "--include-non-clean-photometry",
        action="store_true",
        help="Nao exige p.clean = 1 na fotometria.",
    )
    parser.add_argument(
        "--allow-any-photo-type",
        action="store_true",
        help="Nao exige p.type = 6. Por padrao exige fotometria de estrela.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recria o CSV de saida em vez de retomar.",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="Timeout HTTP para cada consulta.")
    parser.add_argument("--retries", type=int, default=3, help="Tentativas por consulta ao SkyServer.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Pausa entre consultas.")
    return parser.parse_args()


def build_star_candidate_query(
    batch_size: int,
    min_specobjid: int,
    include_zwarning: bool = False,
    include_non_clean_photometry: bool = False,
    allow_any_photo_type: bool = False,
) -> str:
    filters = [
        "s.class = 'STAR'",
        "s.sciencePrimary = 1",
        "s.bestObjID > 0",
        f"s.specObjID > {int(min_specobjid)}",
    ]
    if not include_zwarning:
        filters.append("s.zWarning = 0")
    if not include_non_clean_photometry:
        filters.append("p.clean = 1")
    if not allow_any_photo_type:
        filters.append("p.type = 6")

    where_clause = "\n        AND ".join(filters)
    return f"""
        SELECT TOP {int(batch_size)}
            p.objID AS objID,
            p.ra AS ra,
            p.dec AS dec,
            p.type AS type,
            s.specObjID AS specObjID,
            s.class AS spectralClass,
            s.z AS redshift,
            s.zErr AS redshiftError,
            s.zWarning AS zWarning,
            s.plate AS plate,
            s.mjd AS mjd,
            s.fiberID AS fiberID
        FROM SpecObj AS s
        JOIN PhotoObj AS p
            ON s.bestObjID = p.objID
        WHERE {where_clause}
        ORDER BY s.specObjID ASC
    """


def parse_skyserver_csv(text: str) -> pd.DataFrame:
    content = text.lstrip("\ufeff").strip()
    if not content:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if content.lower().startswith(("error", "exception")):
        raise RuntimeError(content[:1000])

    frame = pd.read_csv(io.StringIO(content), comment="#")
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    if len(frame.columns) == 1:
        first = str(frame.columns[0])
        if "error" in first.lower() or "exception" in first.lower():
            raise RuntimeError(content[:1000])

    column_map = {str(column).strip().lower(): column for column in frame.columns}
    rename = {}
    for expected in OUTPUT_COLUMNS:
        original = column_map.get(expected.lower())
        if original is not None:
            rename[original] = expected

    frame = frame.rename(columns=rename)
    missing = [column for column in OUTPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(
            "Resposta do SkyServer nao trouxe as colunas esperadas: "
            + ", ".join(missing)
        )
    return frame[OUTPUT_COLUMNS].copy()


def request_skyserver_sql(
    session: requests.Session,
    sql: str,
    timeout: float,
    retries: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                SKYSERVER_SQL_URL,
                params={"format": "csv", "cmd": sql},
                timeout=timeout,
            )
            response.raise_for_status()
            return parse_skyserver_csv(response.text)
        except (requests.RequestException, RuntimeError, pd.errors.ParserError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(sleep_seconds)

    if last_error is None:
        raise RuntimeError("Consulta ao SkyServer falhou sem erro detalhado.")
    raise RuntimeError(f"Consulta ao SkyServer falhou: {last_error}") from last_error


def read_existing_output(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    try:
        frame = pd.read_csv(path, dtype={"objID": str, "specObjID": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    for column in OUTPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[OUTPUT_COLUMNS].copy()


def max_specobjid(frame: pd.DataFrame, fallback: int = 0) -> int:
    if frame.empty or "specObjID" not in frame.columns:
        return int(fallback)
    values = pd.to_numeric(frame["specObjID"], errors="coerce").dropna()
    if values.empty:
        return int(fallback)
    return int(values.max())


def append_candidates(path: Path, rows: pd.DataFrame, write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows[OUTPUT_COLUMNS].to_csv(
        path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
    )


def update_identifier_sets(identifiers: dict[str, set[str]], rows: pd.DataFrame) -> None:
    for column in ("objID", "specObjID"):
        if column in rows.columns:
            identifiers.setdefault(column, set()).update(
                value for value in rows[column].map(normalize_identifier) if value
            )


def main() -> int:
    args = parse_args()
    if args.target_candidates < 1:
        raise ValueError("--target-candidates precisa ser maior ou igual a 1.")
    if args.query_batch_size < 1:
        raise ValueError("--query-batch-size precisa ser maior ou igual a 1.")

    root = repo_root()
    output_path = resolve_path(args.output, root)
    dataset_dir = resolve_path(args.dataset_dir, root)

    existing_output = pd.DataFrame(columns=OUTPUT_COLUMNS) if args.overwrite else read_existing_output(output_path)
    saved = len(existing_output)
    min_specobjid = int(args.min_specobjid) if args.min_specobjid is not None else max_specobjid(existing_output)

    if args.overwrite and output_path.exists():
        output_path.unlink()

    identifiers = load_existing_identifiers(
        dataset_dir,
        extra_paths=[] if args.overwrite else [output_path],
    )

    print(f"SkyServer SQL: {SKYSERVER_SQL_URL}")
    print(f"Saida: {output_path}")
    print(f"Candidatos ja salvos: {saved}")
    print(f"Meta de candidatos SDSS novos: {args.target_candidates}")
    print(f"specObjID inicial: {min_specobjid}")

    with requests.Session() as session:
        session.headers.update({"User-Agent": "TCC-SDSS-DR20-star-candidate-query/1.0"})
        while saved < args.target_candidates:
            sql = build_star_candidate_query(
                batch_size=args.query_batch_size,
                min_specobjid=min_specobjid,
                include_zwarning=args.include_zwarning,
                include_non_clean_photometry=args.include_non_clean_photometry,
                allow_any_photo_type=args.allow_any_photo_type,
            )
            raw = request_skyserver_sql(
                session,
                sql,
                timeout=args.timeout,
                retries=args.retries,
                sleep_seconds=args.sleep,
            )

            if raw.empty:
                print("SkyServer nao retornou mais candidatos.")
                break

            min_specobjid = max_specobjid(raw, fallback=min_specobjid)
            remaining = args.target_candidates - saved
            selected, rejected = prepare_sdss_star_candidates(
                raw,
                existing_identifiers=identifiers,
                limit=remaining,
            )

            if not selected.empty:
                write_header = not output_path.exists() or output_path.stat().st_size == 0
                append_candidates(output_path, selected, write_header=write_header)
                update_identifier_sets(identifiers, selected)
                saved += len(selected)

            print(
                "Consulta retornou {raw_count} linhas | novos salvos: {new_count} | "
                "rejeitados/duplicados: {rejected_count} | total salvo: {saved}".format(
                    raw_count=len(raw),
                    new_count=len(selected),
                    rejected_count=len(rejected),
                    saved=saved,
                )
            )

            if len(raw) < args.query_batch_size:
                print("Ultimo lote veio menor que --query-batch-size; busca encerrada.")
                break
            if args.sleep:
                time.sleep(args.sleep)

    print(f"CSV de candidatos pronto: {output_path}")
    print(
        "Proximo passo: python scripts/download_gaia_verified_sdss_stars.py "
        f"--csv {output_path} --target-verified 10000 --dry-run"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        raise SystemExit(2) from None
