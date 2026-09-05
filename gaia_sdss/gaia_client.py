from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from gaia_sdss.config import GAIA_COLUMNS
from gaia_sdss.matching import ERROR, select_best_match


def _table_to_dataframe(table) -> pd.DataFrame:
    return table.to_pandas()


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


def _query_single(row: pd.Series, radius_arcsec: float) -> pd.DataFrame:
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astroquery.gaia import Gaia
    except ImportError as error:
        raise ImportError("astroquery e astropy sao necessarios para consultar Gaia DR3.") from error

    coordinate = SkyCoord(ra=float(row["ra"]) * u.deg, dec=float(row["dec"]) * u.deg)
    radius = radius_arcsec * u.arcsec
    columns = ["source_id", "designation", "ra", "dec"] + [
        column for column in GAIA_COLUMNS if column not in {"source_id", "designation", "ra", "dec"}
    ]
    table = Gaia.query_object_async(
        coordinate=coordinate,
        radius=radius,
        columns=columns,
    )
    df = _table_to_dataframe(table)
    if df.empty:
        return df

    from gaia_sdss.matching import angular_separation_arcsec

    df["separation_arcsec"] = df.apply(
        lambda candidate: angular_separation_arcsec(
            float(row["ra"]),
            float(row["dec"]),
            float(candidate["ra"]),
            float(candidate["dec"]),
        ),
        axis=1,
    )
    return df


def _query_individual_batch(batch: pd.DataFrame, radius_arcsec: float) -> pd.DataFrame:
    candidate_frames = []
    for _, row in batch.iterrows():
        frame = _query_single(row, radius_arcsec)
        if not frame.empty:
            frame["specObjID"] = row["specObjID"]
            frame["objID"] = row.get("objID", "")
        candidate_frames.append(frame)
    return pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()


def _query_batch(rows: pd.DataFrame, radius_arcsec: float) -> pd.DataFrame:
    try:
        from astropy.table import Table
        from astroquery.gaia import Gaia
    except ImportError as error:
        raise ImportError("astroquery e astropy sao necessarios para consultar Gaia DR3.") from error

    upload = Table.from_pandas(
        rows[["specObjID", "objID", "ra", "dec"]].assign(
            local_index=range(len(rows))
        )
    )
    radius_deg = radius_arcsec / 3600.0
    gaia_select = ",\n            ".join(f"g.{column} AS gaia_{column}" for column in GAIA_COLUMNS)
    query = f"""
        SELECT
            sdss.local_index,
            sdss.specObjID,
            sdss.objID,
            sdss.ra AS local_ra,
            sdss.dec AS local_dec,
            {gaia_select},
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
    """
    job = Gaia.launch_job_async(
        query=query,
        upload_resource=upload,
        upload_table_name="sdss_input",
        verbose=False,
    )
    df = _table_to_dataframe(job.get_results())
    df.columns = [column.replace("gaia_", "") if column.startswith("gaia_") else column for column in df.columns]
    df = df.rename(
        columns={
            "specobjid": "specObjID",
            "objid": "objID",
            "separation": "separation_arcsec",
        }
    )
    return df


def match_rows(
    rows: pd.DataFrame,
    cache_path: str | Path,
    radius_arcsec: float,
    ambiguity_delta_arcsec: float,
    batch_size: int,
    force: bool = False,
    retry_errors: bool = False,
    use_batch: bool = True,
) -> pd.DataFrame:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached = pd.DataFrame()
    if cache_path.exists():
        cached = pd.read_csv(cache_path, dtype={"specObjID": str, "objID": str})
        if "specObjID" in cached.columns:
            cached["specObjID"] = cached["specObjID"].astype(str)

    rows = rows.copy()
    rows["specObjID"] = rows["specObjID"].astype(str)
    selected_ids = set(rows["specObjID"])

    if force:
        todo = rows[rows["has_radec"]].copy()
        base_cache = cached[
            ~cached.get("specObjID", pd.Series(dtype=str)).astype(str).isin(selected_ids)
        ].copy() if not cached.empty else cached
    elif retry_errors:
        if cached.empty or "match_status" not in cached.columns:
            todo = rows.iloc[0:0].copy()
            base_cache = cached
        else:
            cached_error_ids = set(
                cached.loc[
                    cached["match_status"].astype(str).str.upper() == ERROR,
                    "specObjID",
                ].astype(str)
            )
            retry_ids = selected_ids & cached_error_ids
            todo = rows[rows["has_radec"] & rows["specObjID"].isin(retry_ids)].copy()
            base_cache = cached[~cached["specObjID"].astype(str).isin(retry_ids)].copy()
    else:
        cached_ids = set(cached.get("specObjID", pd.Series(dtype=str)).astype(str))
        todo = rows[rows["has_radec"] & ~rows["specObjID"].isin(cached_ids)].copy()
        base_cache = cached

    all_new = []
    for start in range(0, len(todo), batch_size):
        batch = todo.iloc[start:start + batch_size].copy()
        if batch.empty:
            continue

        try:
            if use_batch and len(batch) > 1:
                candidates = _query_batch(batch, radius_arcsec)
            else:
                candidates = _query_individual_batch(batch, radius_arcsec)

            for _, row in batch.iterrows():
                local_candidates = candidates[
                    candidates["specObjID"].astype(str) == str(row["specObjID"])
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
            combined_so_far = pd.concat([base_cache, pd.DataFrame(all_new)], ignore_index=True)
            combined_so_far.drop_duplicates("specObjID", keep="last").to_csv(cache_path, index=False)

    combined = pd.concat([base_cache, pd.DataFrame(all_new)], ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates("specObjID", keep="last")
        combined.to_csv(cache_path, index=False)
    return combined
