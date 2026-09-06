from __future__ import annotations

import math
import re

import pandas as pd


EVOLUTION_CLASSES = [
    "WHITE_DWARF",
    "MAIN_SEQUENCE_DWARF",
    "RED_GIANT",
    "GIANT",
    "SUBGIANT",
    "UNKNOWN",
]

ROMAN_LUMINOSITY_CLASSES = ("III", "II", "IV", "V", "I")


def is_missing(value) -> bool:
    if value is None:
        return True
    if pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


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


def parse_sdss_subclass(value) -> dict:
    if is_missing(value):
        raw = "SEM_SUBCLASSE"
    else:
        raw = str(value).strip()

    text = raw.upper()
    if raw == "SEM_SUBCLASSE":
        major = "UNKNOWN"
    elif text.startswith("WD"):
        major = "WD"
    elif text.startswith("CARBON"):
        major = "CARBON"
    else:
        match = re.match(r"^\s*([OBAFGKM])\s*\d?", text)
        major = match.group(1) if match else "UNKNOWN"

    roman = ""
    if major in {"O", "B", "A", "F", "G", "K", "M"}:
        compact = re.sub(r"\s+", "", text)
        compact = re.sub(r"\([^)]*\)", "", compact)
        for candidate in ROMAN_LUMINOSITY_CLASSES:
            if re.search(rf"^[OBAFGKM]\d?{candidate}", compact):
                roman = candidate
                break

    return {
        "spectral_subclass_raw": raw,
        "spectral_class_major": major,
        "roman_luminosity_class_raw": roman,
    }


def absolute_g_mag(phot_g_mean_mag, parallax_mas):
    g_mag = _as_float(phot_g_mean_mag)
    parallax = _as_float(parallax_mas)
    if g_mag is None or parallax is None or parallax <= 0:
        return None
    return g_mag + 5.0 * math.log10(parallax) - 10.0


def parallax_quality(parallax_mas, parallax_error_mas) -> dict:
    parallax = _as_float(parallax_mas)
    error = _as_float(parallax_error_mas)

    if parallax is None:
        return {
            "parallax_quality_flag": "MISSING",
            "parallax_over_error": None,
            "parallax_fractional_error": None,
        }
    if parallax <= 0:
        return {
            "parallax_quality_flag": "INVALID",
            "parallax_over_error": None,
            "parallax_fractional_error": None,
        }
    if error is None or error < 0:
        return {
            "parallax_quality_flag": "UNKNOWN_ERROR",
            "parallax_over_error": None,
            "parallax_fractional_error": None,
        }

    fractional_error = error / parallax
    return {
        "parallax_quality_flag": "GOOD" if fractional_error <= 0.2 else "LOW_CONFIDENCE",
        "parallax_over_error": parallax / error if error > 0 else None,
        "parallax_fractional_error": fractional_error,
    }


def derive_stellar_evolution_class(row: dict) -> dict:
    subclass = parse_sdss_subclass(row.get("pipeline_subclass", row.get("subclass")))
    logg = _as_float(row.get("gaia_logg_gspphot", row.get("logg_gspphot")))
    teff = _as_float(row.get("gaia_teff_gspphot", row.get("teff_gspphot")))
    bp_rp = _as_float(row.get("gaia_bp_rp", row.get("bp_rp")))
    g_mag = row.get("gaia_phot_g_mean_mag", row.get("phot_g_mean_mag"))
    parallax = row.get("gaia_parallax", row.get("parallax"))
    parallax_error_value = row.get("gaia_parallax_error", row.get("parallax_error"))

    absolute_g = absolute_g_mag(g_mag, parallax)
    quality = parallax_quality(parallax, parallax_error_value)

    result = {
        **subclass,
        **quality,
        "absolute_g_mag": absolute_g,
        "stellar_evolution_class": "UNKNOWN",
        "stellar_evolution_method": "",
        "stellar_evolution_status": "UNKNOWN",
    }

    if subclass["spectral_class_major"] == "WD":
        result.update(
            {
                "stellar_evolution_class": "WHITE_DWARF",
                "stellar_evolution_method": "sdss_subclass",
                "stellar_evolution_status": "DERIVED",
            }
        )
        return result

    if logg is not None:
        if logg >= 6.5:
            result.update(
                {
                    "stellar_evolution_class": "WHITE_DWARF",
                    "stellar_evolution_method": "gaia_logg_gspphot",
                    "stellar_evolution_status": "DERIVED",
                }
            )
            return result
        if logg <= 3.5:
            is_red = (bp_rp is not None and bp_rp >= 1.0) or (teff is not None and teff <= 5200.0)
            result.update(
                {
                    "stellar_evolution_class": "RED_GIANT" if is_red else "GIANT",
                    "stellar_evolution_method": "gaia_logg_gspphot",
                    "stellar_evolution_status": "DERIVED",
                }
            )
            return result
        if 3.5 < logg < 4.1:
            result.update(
                {
                    "stellar_evolution_class": "SUBGIANT",
                    "stellar_evolution_method": "gaia_logg_gspphot",
                    "stellar_evolution_status": "DERIVED",
                }
            )
            return result

    if absolute_g is not None and bp_rp is not None:
        if bp_rp < 0.8 and absolute_g > 9.0:
            result.update(
                {
                    "stellar_evolution_class": "WHITE_DWARF",
                    "stellar_evolution_method": "gaia_hr_diagram",
                    "stellar_evolution_status": "DERIVED",
                }
            )
            return result
        if bp_rp >= 1.0 and absolute_g <= 2.5:
            result.update(
                {
                    "stellar_evolution_class": "RED_GIANT",
                    "stellar_evolution_method": "gaia_hr_diagram",
                    "stellar_evolution_status": "DERIVED",
                }
            )
            return result
        if absolute_g <= 1.5:
            result.update(
                {
                    "stellar_evolution_class": "GIANT",
                    "stellar_evolution_method": "gaia_hr_diagram",
                    "stellar_evolution_status": "DERIVED",
                }
            )
            return result
        if 1.5 < absolute_g <= 3.5:
            result.update(
                {
                    "stellar_evolution_class": "SUBGIANT",
                    "stellar_evolution_method": "gaia_hr_diagram",
                    "stellar_evolution_status": "DERIVED",
                }
            )
            return result
        result.update(
            {
                "stellar_evolution_class": "MAIN_SEQUENCE_DWARF",
                "stellar_evolution_method": "gaia_hr_diagram",
                "stellar_evolution_status": "DERIVED",
            }
        )
        return result

    if quality["parallax_quality_flag"] == "MISSING":
        result["stellar_evolution_status"] = "MISSING_GAIA_PARALLAX"
    elif quality["parallax_quality_flag"] == "INVALID":
        result["stellar_evolution_status"] = "INVALID_GAIA_PARALLAX"
    else:
        result["stellar_evolution_status"] = "INSUFFICIENT_GAIA_DATA"
    return result
