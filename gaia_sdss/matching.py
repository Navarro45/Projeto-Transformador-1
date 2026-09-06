from __future__ import annotations

import math
from typing import Iterable

import pandas as pd

from gaia_sdss.config import TEFF_SPECTRAL_LIMITS


MATCH_UNIQUE = "MATCH_UNIQUE"
MATCH_BEST = "MATCH_BEST"
MATCH_AMBIGUOUS = "MATCH_AMBIGUOUS"
NO_MATCH = "NO_MATCH"
ERROR = "ERROR"


def angular_separation_arcsec(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    ra1 = math.radians(float(ra1_deg))
    dec1 = math.radians(float(dec1_deg))
    ra2 = math.radians(float(ra2_deg))
    dec2 = math.radians(float(dec2_deg))

    sin_ddec = math.sin((dec2 - dec1) / 2.0)
    sin_dra = math.sin((ra2 - ra1) / 2.0)
    a = sin_ddec * sin_ddec + math.cos(dec1) * math.cos(dec2) * sin_dra * sin_dra
    a = min(1.0, max(0.0, a))
    return math.degrees(2.0 * math.asin(math.sqrt(a))) * 3600.0


def _as_records(candidates) -> list[dict]:
    if candidates is None:
        return []
    if isinstance(candidates, pd.DataFrame):
        return candidates.to_dict("records")
    return list(candidates)


def select_best_match(
    candidates,
    radius_arcsec: float,
    ambiguity_delta_arcsec: float,
) -> dict:
    records = _as_records(candidates)
    records = [
        row for row in records
        if row.get("separation_arcsec") is not None and pd.notna(row.get("separation_arcsec"))
    ]
    records.sort(key=lambda row: float(row["separation_arcsec"]))

    if not records:
        return {
            "match_status": NO_MATCH,
            "candidate_count": 0,
            "best_distance_arcsec": None,
            "second_best_distance_arcsec": None,
            "distance_gap_arcsec": None,
            "is_ambiguous": False,
            "best": {},
        }

    best = records[0]
    if float(best["separation_arcsec"]) > radius_arcsec:
        return {
            "match_status": NO_MATCH,
            "candidate_count": len(records),
            "best_distance_arcsec": float(best["separation_arcsec"]),
            "second_best_distance_arcsec": None,
            "distance_gap_arcsec": None,
            "is_ambiguous": False,
            "best": {},
        }

    second = records[1] if len(records) > 1 else None
    second_distance = float(second["separation_arcsec"]) if second else None
    gap = None if second_distance is None else second_distance - float(best["separation_arcsec"])

    if len(records) == 1:
        status = MATCH_UNIQUE
        ambiguous = False
    elif gap is not None and gap <= ambiguity_delta_arcsec:
        status = MATCH_AMBIGUOUS
        ambiguous = True
    else:
        status = MATCH_BEST
        ambiguous = False

    return {
        "match_status": status,
        "candidate_count": len(records),
        "best_distance_arcsec": float(best["separation_arcsec"]),
        "second_best_distance_arcsec": second_distance,
        "distance_gap_arcsec": gap,
        "is_ambiguous": ambiguous,
        "best": best,
    }


def derive_spectral_class_from_teff(teff) -> dict:
    if teff is None or pd.isna(teff):
        return {"gaia_derived_class": "", "gaia_derived_method": "teff_gspphot", "gaia_derived_status": "MISSING"}

    try:
        value = float(teff)
    except (TypeError, ValueError):
        return {"gaia_derived_class": "", "gaia_derived_method": "teff_gspphot", "gaia_derived_status": "INVALID"}

    if value <= 0:
        return {"gaia_derived_class": "", "gaia_derived_method": "teff_gspphot", "gaia_derived_status": "INVALID"}

    for spectral_class, lower, upper in TEFF_SPECTRAL_LIMITS:
        if lower <= value < upper:
            return {
                "gaia_derived_class": spectral_class,
                "gaia_derived_method": "teff_gspphot",
                "gaia_derived_status": "DERIVED",
            }

    return {"gaia_derived_class": "", "gaia_derived_method": "teff_gspphot", "gaia_derived_status": "OUT_OF_RANGE"}


def major_sdss_subclass(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if not text or text == "SEM_SUBCLASSE":
        return ""
    for spectral_class in ("O", "B", "A", "F", "G", "K", "M"):
        if text.startswith(spectral_class):
            return spectral_class
    return ""


def derive_gaia_dsc_class(row: dict) -> dict:
    probabilities = {
        "STAR": row.get("classprob_dsc_combmod_star"),
        "QSO": row.get("classprob_dsc_combmod_quasar"),
        "GALAXY": row.get("classprob_dsc_combmod_galaxy"),
    }

    valid = {}
    for label, value in probabilities.items():
        if value is None or pd.isna(value):
            continue
        try:
            valid[label] = float(value)
        except (TypeError, ValueError):
            continue

    if not valid:
        return {"gaia_dsc_class": "", "gaia_dsc_probability": None, "gaia_dsc_status": "MISSING"}

    label, probability = max(valid.items(), key=lambda item: item[1])
    return {"gaia_dsc_class": label, "gaia_dsc_probability": probability, "gaia_dsc_status": "DERIVED"}


def compare_classes(row: dict) -> dict:
    pipeline_class = str(row.get("spectralClass", "") or "").upper()
    pipeline_subclass_major = major_sdss_subclass(row.get("pipeline_subclass"))
    gaia_derived_class = str(row.get("gaia_derived_class", "") or "").upper()
    gaia_dsc_class = str(row.get("gaia_dsc_class", "") or "").upper()

    if pipeline_class == "STAR" and pipeline_subclass_major and gaia_derived_class:
        spectral_status = "AGREE" if pipeline_subclass_major == gaia_derived_class else "DISAGREE"
    elif pipeline_class == "STAR":
        spectral_status = "NOT_COMPARABLE"
    else:
        spectral_status = "NOT_STELLAR_SDSS_CLASS"

    if pipeline_class and gaia_dsc_class:
        dsc_status = "AGREE" if pipeline_class == gaia_dsc_class else "DISAGREE"
    else:
        dsc_status = "NOT_COMPARABLE"

    return {
        "pipeline_subclass_major": pipeline_subclass_major,
        "spectral_agreement_status": spectral_status,
        "dsc_agreement_status": dsc_status,
    }


def proper_motion_total(pmra, pmdec):
    if pmra is None or pmdec is None or pd.isna(pmra) or pd.isna(pmdec):
        return None
    try:
        return math.hypot(float(pmra), float(pmdec))
    except (TypeError, ValueError):
        return None
