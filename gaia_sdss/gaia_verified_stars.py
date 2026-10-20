from __future__ import annotations

import math
import re
import time
import io
import shutil
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from gaia_sdss.config import GAIA_COLUMNS
from gaia_sdss.matching import (
    ERROR,
    MATCH_AMBIGUOUS,
    MATCH_BEST,
    MATCH_UNIQUE,
    NO_MATCH,
    select_best_match,
)


GAIA_AP_COLUMNS = [
    "spectraltype_esphs",
    "flags_esphs",
    "lum_flame",
    "lum_flame_lower",
    "lum_flame_upper",
    "evolstage_flame",
    "radius_flame",
    "teff_esphs",
    "logg_esphs",
]

REQUIRED_SDSS_COLUMNS = {"objID", "specObjID", "spectralClass", "ra", "dec"}
VALID_DOWNLOAD_MATCH_STATUSES = {MATCH_UNIQUE, MATCH_BEST}
VALID_AMBIGUOUS_DOWNLOAD_MATCH_STATUSES = VALID_DOWNLOAD_MATCH_STATUSES | {MATCH_AMBIGUOUS}
GAIA_TAP_SYNC_URL = "https://gea.esac.esa.int/tap-server/tap/sync"


def normalize_identifier(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def is_missing(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def _as_float(value):
    if is_missing(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _reject_rows(rows: pd.DataFrame, reason: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=list(rows.columns) + ["rejection_reason"])
    rejected = rows.copy()
    rejected["rejection_reason"] = reason
    return rejected


def read_sdss_candidate_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV SDSS nao encontrado: {path}")
    try:
        df = pd.read_csv(path, dtype={"objID": str, "specObjID": str})
    except pd.errors.EmptyDataError as error:
        raise ValueError(
            "CSV SDSS esta vazio. Informe um CSV preenchido com as colunas "
            "objID, specObjID, spectralClass, ra e dec."
        ) from error
    df.columns = [str(column).strip() for column in df.columns]
    missing = REQUIRED_SDSS_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV SDSS precisa conter as colunas: {', '.join(sorted(missing))}")
    return df


def load_existing_identifiers(dataset_dir: str | Path, extra_paths: list[str | Path] | None = None) -> dict[str, set[str]]:
    dataset_dir = Path(dataset_dir)
    paths = [
        dataset_dir / "manifests" / "download_manifest.csv",
        dataset_dir / "metadata" / "spectra_metadata.csv",
        dataset_dir / "gaia_sdss" / "cache" / "gaia_matches.csv",
        dataset_dir / "gaia_sdss" / "gaia_sdss_comparison.csv",
    ]
    if extra_paths:
        paths.extend(Path(path) for path in extra_paths)

    identifiers = {"objID": set(), "specObjID": set(), "source_id": set()}
    for path in paths:
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(
                path,
                usecols=lambda column: column in {"objID", "specObjID", "source_id", "gaia_source_id"},
                dtype=str,
            )
        except (ValueError, pd.errors.EmptyDataError):
            continue

        for column in ("objID", "specObjID"):
            if column in frame.columns:
                identifiers[column].update(
                    value for value in frame[column].map(normalize_identifier) if value
                )

        for column in ("source_id", "gaia_source_id"):
            if column in frame.columns:
                identifiers["source_id"].update(
                    value for value in frame[column].map(normalize_identifier) if value
                )

    return identifiers


def prepare_sdss_star_candidates(
    rows: pd.DataFrame,
    existing_identifiers: dict[str, set[str]] | None = None,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing_identifiers = existing_identifiers or {"objID": set(), "specObjID": set(), "source_id": set()}
    working = rows.copy()
    working.columns = [str(column).strip() for column in working.columns]
    missing = REQUIRED_SDSS_COLUMNS - set(working.columns)
    if missing:
        raise ValueError(f"CSV SDSS precisa conter as colunas: {', '.join(sorted(missing))}")

    working["_source_row"] = range(len(working))
    working["objID"] = working["objID"].map(normalize_identifier)
    working["specObjID"] = working["specObjID"].map(normalize_identifier)
    working["spectralClass"] = working["spectralClass"].fillna("").astype(str).str.strip().str.upper()
    working["ra"] = pd.to_numeric(working["ra"], errors="coerce")
    working["dec"] = pd.to_numeric(working["dec"], errors="coerce")
    working["has_radec"] = working["ra"].notna() & working["dec"].notna()

    rejected_parts = []
    active = working

    mask = (active["objID"] == "") | (active["specObjID"] == "") | ~active["has_radec"]
    rejected_parts.append(_reject_rows(active[mask], "missing_required_sdss_identifier_or_position"))
    active = active[~mask].copy()

    mask = active["spectralClass"] != "STAR"
    rejected_parts.append(_reject_rows(active[mask], "not_sdss_star"))
    active = active[~mask].copy()

    existing_obj_ids = existing_identifiers.get("objID", set())
    existing_spec_ids = existing_identifiers.get("specObjID", set())
    mask = active["objID"].isin(existing_obj_ids) | active["specObjID"].isin(existing_spec_ids)
    rejected_parts.append(_reject_rows(active[mask], "already_exists_in_local_dataset"))
    active = active[~mask].copy()

    mask = active["specObjID"].duplicated(keep="first")
    rejected_parts.append(_reject_rows(active[mask], "duplicate_specObjID_in_input"))
    active = active[~mask].copy()

    mask = active["objID"].duplicated(keep="first")
    rejected_parts.append(_reject_rows(active[mask], "duplicate_objID_in_input"))
    active = active[~mask].copy()

    active = active.sort_values("_source_row").reset_index(drop=True)
    if limit is not None:
        active = active.head(limit).copy()

    rejected = pd.concat(
        [part for part in rejected_parts if not part.empty],
        ignore_index=True,
    ) if any(not part.empty for part in rejected_parts) else pd.DataFrame(columns=list(working.columns) + ["rejection_reason"])

    return active, rejected


def parse_gaia_spectral_class(value) -> str:
    if is_missing(value):
        return ""
    text = str(value).strip().upper()
    if text == "CSTAR":
        return "CSTAR"
    match = re.match(r"^\s*([OBAFGKM])", text)
    return match.group(1) if match else ""


def gaia_luminosity_class_from_evolstage(value) -> str:
    parsed = _as_float(value)
    if parsed is None:
        return ""

    stage = int(round(parsed))
    if stage <= 0:
        return ""
    if stage <= 420:
        return "MAIN_SEQUENCE"
    if stage < 490:
        return "SUBGIANT"
    if stage < 860:
        return "RED_GIANT_BRANCH"
    if stage < 1290:
        return "RGB_BUMP"
    return "RGB_TIP"


def annotate_gaia_required_fields(matches: pd.DataFrame) -> pd.DataFrame:
    annotated = matches.copy()
    if annotated.empty:
        return annotated

    annotated["gaia_spectral_class"] = annotated.get(
        "spectraltype_esphs",
        pd.Series([""] * len(annotated), index=annotated.index),
    ).map(parse_gaia_spectral_class)
    annotated["gaia_luminosity_class"] = annotated.get(
        "evolstage_flame",
        pd.Series([""] * len(annotated), index=annotated.index),
    ).map(gaia_luminosity_class_from_evolstage)
    annotated["has_gaia_spectral_class"] = annotated["gaia_spectral_class"] != ""
    annotated["has_gaia_luminosity_class"] = annotated["gaia_luminosity_class"] != ""
    if "lum_flame" in annotated.columns:
        annotated["has_gaia_luminosity_value"] = annotated["lum_flame"].map(lambda value: not is_missing(value))
    else:
        annotated["has_gaia_luminosity_value"] = False
    annotated["has_required_gaia_ap"] = (
        annotated["has_gaia_spectral_class"]
        & annotated["has_gaia_luminosity_class"]
        & annotated["has_gaia_luminosity_value"]
    )

    status = []
    for _, row in annotated.iterrows():
        if row["has_required_gaia_ap"]:
            status.append("OK")
        elif (
            not row["has_gaia_spectral_class"]
            and (not row["has_gaia_luminosity_class"] or not row["has_gaia_luminosity_value"])
        ):
            status.append("MISSING_GAIA_SPECTRAL_AND_LUMINOSITY_INFO")
        elif not row["has_gaia_spectral_class"]:
            status.append("MISSING_GAIA_SPECTRAL_CLASS")
        elif not row["has_gaia_luminosity_class"]:
            status.append("MISSING_GAIA_LUMINOSITY_CLASS")
        else:
            status.append("MISSING_GAIA_LUMINOSITY_VALUE")
    annotated["gaia_required_status"] = status
    return annotated


def filter_gaia_verified_candidates(
    candidates: pd.DataFrame,
    matches: pd.DataFrame,
    existing_gaia_source_ids: set[str] | None = None,
    allow_ambiguous: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing_gaia_source_ids = existing_gaia_source_ids or set()
    gaia = annotate_gaia_required_fields(matches)

    base = candidates.copy()
    base["specObjID"] = base["specObjID"].map(normalize_identifier)
    if gaia.empty:
        gaia = pd.DataFrame(columns=["specObjID", "match_status"])
    gaia["specObjID"] = gaia["specObjID"].map(normalize_identifier)

    merged = base.merge(gaia, on="specObjID", how="left", suffixes=("", "_gaia"))
    if "objID_gaia" in merged.columns:
        merged = merged.drop(columns=["objID_gaia"])

    allowed = VALID_AMBIGUOUS_DOWNLOAD_MATCH_STATUSES if allow_ambiguous else VALID_DOWNLOAD_MATCH_STATUSES
    rejected_parts = []

    match_status = merged.get("match_status", pd.Series([""] * len(merged), index=merged.index)).fillna("")
    mask = ~match_status.isin(allowed)
    rejected_parts.append(_reject_rows(merged[mask], "gaia_match_not_unique_enough"))
    active = merged[~mask].copy()

    required = active.get("has_required_gaia_ap", pd.Series([False] * len(active), index=active.index)).fillna(False)
    mask = ~required
    rejected = _reject_rows(active[mask], "missing_required_gaia_spectral_or_luminosity_class")
    if not rejected.empty and "gaia_required_status" in rejected.columns:
        rejected["rejection_reason"] = rejected["gaia_required_status"].fillna(rejected["rejection_reason"])
    rejected_parts.append(rejected)
    active = active[~mask].copy()

    if "source_id" in active.columns:
        active["_source_id_norm"] = active["source_id"].map(normalize_identifier)
        mask = active["_source_id_norm"].isin(existing_gaia_source_ids)
        rejected_parts.append(_reject_rows(active[mask], "gaia_source_already_exists_in_local_outputs"))
        active = active[~mask].copy()

        active = active.sort_values(["_source_id_norm", "best_distance_arcsec"], na_position="last")
        mask = active["_source_id_norm"].duplicated(keep="first") & (active["_source_id_norm"] != "")
        rejected_parts.append(_reject_rows(active[mask], "duplicate_gaia_source_id_in_verified_batch"))
        active = active[~mask].drop(columns=["_source_id_norm"]).copy()

    active = active.sort_values("_source_row").reset_index(drop=True)
    rejected = pd.concat(
        [part for part in rejected_parts if not part.empty],
        ignore_index=True,
    ) if any(not part.empty for part in rejected_parts) else pd.DataFrame(columns=list(merged.columns) + ["rejection_reason"])
    return active, rejected


def _table_to_dataframe(table) -> pd.DataFrame:
    return table.to_pandas()


def _normalize_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "specobjid": "specObjID",
        "objid": "objID",
        "separation": "separation_arcsec",
    }
    return df.rename(columns={column: rename.get(str(column).lower(), column) for column in df.columns})


def _empty_match(local_row: pd.Series, status: str, error: str = "") -> dict:
    return {
        "specObjID": str(local_row.get("specObjID", "")),
        "objID": str(local_row.get("objID", "")),
        "match_status": status,
        "candidate_count": 0,
        "best_distance_arcsec": None,
        "second_best_distance_arcsec": None,
        "distance_gap_arcsec": None,
        "is_ambiguous": False,
        "gaia_query_time_utc": datetime.now(timezone.utc).isoformat(),
        "gaia_error": error,
    }


def _concat_cache_frames(base_cache: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    if base_cache.empty:
        return new_rows.copy()
    if new_rows.empty:
        return base_cache.copy()
    return pd.concat([base_cache, new_rows], ignore_index=True)


def _cache_backup_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.name}.bak")


def _read_cache_csv(cache_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(cache_path, dtype={"specObjID": str, "objID": str, "source_id": str})
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        backup_path = _cache_backup_path(cache_path)
        if backup_path.exists() and backup_path.stat().st_size > 0:
            return pd.read_csv(backup_path, dtype={"specObjID": str, "objID": str, "source_id": str})
        raise


def _write_cache_csv_atomic(cache_path: Path, frame: pd.DataFrame) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    backup_path = _cache_backup_path(cache_path)

    frame.to_csv(tmp_path, index=False)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        shutil.copy2(cache_path, backup_path)
    tmp_path.replace(cache_path)


def _select_sql_columns() -> str:
    source_columns = ",\n            ".join(f"g.{column} AS {column}" for column in GAIA_COLUMNS)
    ap_columns = ",\n            ".join(f"ap.{column} AS {column}" for column in GAIA_AP_COLUMNS)
    return f"{source_columns},\n            {ap_columns}"


def _query_batch_with_ap(rows: pd.DataFrame, radius_arcsec: float) -> pd.DataFrame:
    try:
        from astropy.table import Table
        from astroquery.gaia import Gaia
    except ImportError as error:
        raise ImportError("astroquery e astropy sao necessarios para consultar Gaia DR3.") from error

    upload = Table.from_pandas(
        rows[["specObjID", "objID", "ra", "dec"]].assign(local_index=range(len(rows)))
    )
    radius_deg = radius_arcsec / 3600.0
    query = f"""
        SELECT
            sdss.local_index,
            sdss.specObjID,
            sdss.objID,
            sdss.ra AS local_ra,
            sdss.dec AS local_dec,
            {_select_sql_columns()},
            DISTANCE(
                POINT('ICRS', sdss.ra, sdss.dec),
                POINT('ICRS', g.ra, g.dec)
            ) * 3600.0 AS separation_arcsec
        FROM TAP_UPLOAD.sdss_input AS sdss
        JOIN gaiadr3.gaia_source AS g
        ON 1 = CONTAINS(
            POINT('ICRS', g.ra, g.dec),
            CIRCLE('ICRS', sdss.ra, sdss.dec, {radius_deg:.12f})
        )
        LEFT JOIN gaiadr3.astrophysical_parameters AS ap
        ON ap.source_id = g.source_id
    """
    job = Gaia.launch_job_async(
        query=query,
        upload_resource=upload,
        upload_table_name="sdss_input",
        verbose=False,
    )
    return _normalize_result_columns(_table_to_dataframe(job.get_results()))


def _request_gaia_tap_csv(query: str, timeout_seconds: float) -> pd.DataFrame:
    response = requests.post(
        GAIA_TAP_SYNC_URL,
        data={
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "csv",
            "QUERY": query,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()

    text = response.text.lstrip("\ufeff").strip()
    if not text:
        return pd.DataFrame()
    if text.startswith("<") or text.lower().startswith(("error", "exception")):
        raise RuntimeError(text[:1000])
    return _normalize_result_columns(pd.read_csv(io.StringIO(text), dtype={"source_id": str}))


def _query_ap_for_source_ids(source_ids: list[str], timeout_seconds: float) -> pd.DataFrame:
    clean_ids = [normalize_identifier(value) for value in source_ids if normalize_identifier(value)]
    if not clean_ids:
        return pd.DataFrame(columns=["source_id"] + GAIA_AP_COLUMNS)

    id_list = ", ".join(clean_ids)
    ap_select = ",\n            ".join(f"ap.{column} AS {column}" for column in GAIA_AP_COLUMNS)
    query = f"""
        SELECT
            ap.source_id AS source_id,
            {ap_select}
        FROM gaiadr3.astrophysical_parameters AS ap
        WHERE ap.source_id IN ({id_list})
    """
    return _request_gaia_tap_csv(query, timeout_seconds)


def _query_single_with_ap(row: pd.Series, radius_arcsec: float, timeout_seconds: float = 60.0) -> pd.DataFrame:
    radius_deg = radius_arcsec / 3600.0
    ra = float(row["ra"])
    dec = float(row["dec"])
    source_select = ",\n            ".join(f"g.{column} AS {column}" for column in GAIA_COLUMNS)
    query = f"""
        SELECT TOP 100
            {source_select},
            DISTANCE(
                POINT('ICRS', {ra:.12f}, {dec:.12f}),
                POINT('ICRS', g.ra, g.dec)
            ) * 3600.0 AS separation_arcsec
        FROM gaiadr3.gaia_source AS g
        WHERE 1 = CONTAINS(
            POINT('ICRS', g.ra, g.dec),
            CIRCLE('ICRS', {ra:.12f}, {dec:.12f}, {radius_deg:.12f})
        )
        ORDER BY separation_arcsec ASC
    """
    df = _request_gaia_tap_csv(query, timeout_seconds)

    if not df.empty and "source_id" in df.columns:
        df["source_id"] = df["source_id"].map(normalize_identifier)
        ap = _query_ap_for_source_ids(df["source_id"].tolist(), timeout_seconds)
        if not ap.empty and "source_id" in ap.columns:
            ap["source_id"] = ap["source_id"].map(normalize_identifier)
            df = df.merge(ap, on="source_id", how="left")

    if not df.empty:
        df["specObjID"] = row["specObjID"]
        df["objID"] = row.get("objID", "")
    return df


def _query_individual_batch_with_ap(batch: pd.DataFrame, radius_arcsec: float) -> pd.DataFrame:
    frames, errors = _query_individual_candidates_with_errors(
        batch,
        radius_arcsec,
        query_retries=1,
        retry_sleep_seconds=0.0,
        network_timeout=60.0,
    )
    if errors:
        error_preview = "; ".join(f"{key}: {value}" for key, value in list(errors.items())[:3])
        raise RuntimeError(f"Falha em consulta individual Gaia: {error_preview}")
    return frames


def _run_with_retries(action, query_retries: int, retry_sleep_seconds: float):
    attempts = max(1, int(query_retries))
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt < attempts and retry_sleep_seconds > 0:
                time.sleep(retry_sleep_seconds)
    raise last_error


def _query_individual_candidates_with_errors(
    batch: pd.DataFrame,
    radius_arcsec: float,
    query_retries: int,
    retry_sleep_seconds: float,
    network_timeout: float,
) -> tuple[pd.DataFrame, dict[str, str]]:
    frames = []
    errors: dict[str, str] = {}
    for _, row in batch.iterrows():
        spec_obj_id = normalize_identifier(row.get("specObjID", ""))
        try:
            frame = _run_with_retries(
                lambda row=row: _query_single_with_ap(
                    row,
                    radius_arcsec,
                    timeout_seconds=network_timeout,
                ),
                query_retries=query_retries,
                retry_sleep_seconds=retry_sleep_seconds,
            )
            if not frame.empty:
                frames.append(frame)
        except Exception as error:  # noqa: BLE001
            errors[spec_obj_id] = f"{type(error).__name__}: {error}"

    candidates = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return candidates, errors


def match_rows_with_gaia_ap(
    rows: pd.DataFrame,
    cache_path: str | Path,
    radius_arcsec: float,
    ambiguity_delta_arcsec: float,
    batch_size: int,
    force: bool = False,
    retry_errors: bool = False,
    use_batch: bool = True,
    query_retries: int = 3,
    retry_sleep_seconds: float = 5.0,
    network_timeout: float = 60.0,
    fallback_to_individual: bool = True,
) -> pd.DataFrame:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached = pd.DataFrame()
    if cache_path.exists():
        cached = _read_cache_csv(cache_path)
        if "specObjID" in cached.columns:
            cached["specObjID"] = cached["specObjID"].map(normalize_identifier)

    rows = rows.copy()
    rows["specObjID"] = rows["specObjID"].map(normalize_identifier)
    selected_ids = set(rows["specObjID"])

    if force:
        todo = rows[rows["has_radec"]].copy()
        base_cache = cached[
            ~cached.get("specObjID", pd.Series(dtype=str)).map(normalize_identifier).isin(selected_ids)
        ].copy() if not cached.empty else cached
    elif retry_errors:
        if cached.empty or "match_status" not in cached.columns:
            todo = rows[rows["has_radec"]].copy()
            base_cache = cached
        else:
            cached_ids = set(cached.get("specObjID", pd.Series(dtype=str)).map(normalize_identifier))
            cached_error_ids = set(
                cached.loc[
                    cached["match_status"].astype(str).str.upper() == ERROR,
                    "specObjID",
                ].map(normalize_identifier)
            )
            retry_ids = selected_ids & cached_error_ids
            new_ids = selected_ids - cached_ids
            todo_ids = retry_ids | new_ids
            todo = rows[rows["has_radec"] & rows["specObjID"].isin(todo_ids)].copy()
            base_cache = cached[~cached["specObjID"].isin(retry_ids)].copy()
    else:
        cached_ids = set(cached.get("specObjID", pd.Series(dtype=str)).map(normalize_identifier))
        todo = rows[rows["has_radec"] & ~rows["specObjID"].isin(cached_ids)].copy()
        base_cache = cached

    all_new = []
    for start in range(0, len(todo), batch_size):
        batch = todo.iloc[start:start + batch_size].copy()
        if batch.empty:
            continue

        try:
            row_errors: dict[str, str] = {}
            if use_batch and len(batch) > 1:
                try:
                    candidates = _run_with_retries(
                        lambda: _query_batch_with_ap(batch, radius_arcsec),
                        query_retries=query_retries,
                        retry_sleep_seconds=retry_sleep_seconds,
                    )
                except Exception:
                    if not fallback_to_individual:
                        raise
                    candidates, row_errors = _query_individual_candidates_with_errors(
                        batch,
                        radius_arcsec,
                        query_retries=query_retries,
                        retry_sleep_seconds=retry_sleep_seconds,
                        network_timeout=network_timeout,
                    )
            else:
                candidates, row_errors = _query_individual_candidates_with_errors(
                    batch,
                    radius_arcsec,
                    query_retries=query_retries,
                    retry_sleep_seconds=retry_sleep_seconds,
                    network_timeout=network_timeout,
                )

            for _, row in batch.iterrows():
                spec_obj_id = normalize_identifier(row.get("specObjID", ""))
                if spec_obj_id in row_errors:
                    all_new.append(_empty_match(row, ERROR, row_errors[spec_obj_id]))
                    continue

                local_candidates = candidates[
                    candidates["specObjID"].map(normalize_identifier) == spec_obj_id
                ] if not candidates.empty and "specObjID" in candidates.columns else pd.DataFrame()
                selected = select_best_match(local_candidates, radius_arcsec, ambiguity_delta_arcsec)
                output = _empty_match(row, selected["match_status"])
                output.update(
                    {
                        "candidate_count": selected["candidate_count"],
                        "best_distance_arcsec": selected["best_distance_arcsec"],
                        "second_best_distance_arcsec": selected["second_best_distance_arcsec"],
                        "distance_gap_arcsec": selected["distance_gap_arcsec"],
                        "is_ambiguous": selected["is_ambiguous"],
                    }
                )
                for key, value in selected["best"].items():
                    if key not in {"specObjID", "objID", "local_index", "local_ra", "local_dec"}:
                        output[key] = value
                all_new.append(output)

        except Exception as error:  # noqa: BLE001
            for _, row in batch.iterrows():
                all_new.append(_empty_match(row, ERROR, f"{type(error).__name__}: {error}"))

        if all_new:
            combined_so_far = _concat_cache_frames(base_cache, pd.DataFrame(all_new))
            combined_so_far = combined_so_far.drop_duplicates("specObjID", keep="last")
            _write_cache_csv_atomic(cache_path, combined_so_far)

    combined = _concat_cache_frames(base_cache, pd.DataFrame(all_new))
    if not combined.empty:
        combined = combined.drop_duplicates("specObjID", keep="last")
        _write_cache_csv_atomic(cache_path, combined)
    return combined
