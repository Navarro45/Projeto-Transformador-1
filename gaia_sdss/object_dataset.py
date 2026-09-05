from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from gaia_sdss.stellar import derive_stellar_evolution_class, is_missing


def _json_value(value):
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _text_value(value, default: str = "") -> str:
    if is_missing(value):
        return default
    return str(value)


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_ready(inner) for inner in value]
    return _json_value(value)


def _compact_dict(row: dict, keys: list[str]) -> dict:
    return {key: _json_value(row.get(key)) for key in keys if key in row and not is_missing(row.get(key))}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _existing_path(value) -> Path | None:
    if is_missing(value):
        return None
    path = Path(str(value))
    return path if path.exists() else None


def _copy_asset(source: Path | None, destination: Path, copy_assets: bool) -> str:
    if source is None:
        return ""
    if not copy_assets:
        return ""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def _derive_plot_path(row: dict, dataset_root: Path) -> Path | None:
    spec_obj_id = str(row.get("specObjID", "")).strip()
    if not spec_obj_id:
        return None

    candidates = [
        row.get("spectrum_plot_path"),
        dataset_root / "plots" / "star" / f"{spec_obj_id}_spectrum.png",
        dataset_root / "plots" / f"{spec_obj_id}_spectrum.png",
    ]
    for candidate in candidates:
        path = _existing_path(candidate)
        if path is not None:
            return path
    return None


def _ensure_stellar_fields(row: dict) -> dict:
    required = {
        "spectral_subclass_raw",
        "spectral_class_major",
        "roman_luminosity_class_raw",
        "stellar_evolution_class",
        "stellar_evolution_method",
        "stellar_evolution_status",
        "absolute_g_mag",
        "parallax_quality_flag",
    }
    if required.issubset(row.keys()):
        return row
    updated = dict(row)
    updated.update(derive_stellar_evolution_class(updated))
    return updated


def build_star_object_dataset(
    comparison_csv: str | Path,
    output_dir: str | Path,
    dataset_root: str | Path,
    copy_assets: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    comparison_csv = Path(comparison_csv)
    output_dir = Path(output_dir)
    dataset_root = Path(dataset_root)

    if not comparison_csv.exists():
        raise FileNotFoundError(f"Dataset consolidado nao encontrado: {comparison_csv}")

    df = pd.read_csv(comparison_csv, dtype={"objID": str, "specObjID": str})
    if "spectralClass" not in df.columns:
        raise ValueError("Dataset consolidado precisa conter spectralClass.")

    stars = df[df["spectralClass"].fillna("").astype(str).str.upper() == "STAR"].copy()
    if limit is not None:
        stars = stars.head(limit).copy()

    index_rows = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for sequential_id, (_, record) in enumerate(stars.iterrows(), start=1):
        row = _ensure_stellar_fields(record.to_dict())
        star_id = f"STAR_{sequential_id:06d}"
        object_dir = output_dir / star_id

        image_path = _existing_path(row.get("image_path"))
        spectrum_path = _existing_path(row.get("spectrum_path"))
        spectrum_metadata_path = _existing_path(row.get("spectrum_metadata_path"))
        spectrum_plot_path = _derive_plot_path(row, dataset_root)

        copied_image = _copy_asset(
            image_path,
            object_dir / "RGB" / f"image_rgb{image_path.suffix if image_path else '.jpg'}",
            copy_assets,
        )
        copied_plot = _copy_asset(
            spectrum_plot_path,
            object_dir / "Spectrogram" / f"spectrum_plot{spectrum_plot_path.suffix if spectrum_plot_path else '.png'}",
            copy_assets,
        )

        target = {
            "spectral_class": _text_value(row.get("spectralClass"), "STAR"),
            "spectral_subclass_raw": _text_value(row.get("spectral_subclass_raw", row.get("pipeline_subclass")), "SEM_SUBCLASSE"),
            "spectral_class_major": _text_value(row.get("spectral_class_major"), "UNKNOWN"),
            "roman_luminosity_class_raw": _text_value(row.get("roman_luminosity_class_raw"), ""),
            "stellar_evolution_class": _text_value(row.get("stellar_evolution_class"), "UNKNOWN"),
            "stellar_evolution_method": _text_value(row.get("stellar_evolution_method"), ""),
            "stellar_evolution_status": _text_value(row.get("stellar_evolution_status"), "UNKNOWN"),
        }
        rgb = {
            "image_path": str(image_path) if image_path else "",
            "copied_image_path": copied_image,
            "copy_assets": copy_assets,
        }
        spectrogram = {
            "spectrum_path": str(spectrum_path) if spectrum_path else "",
            "spectrum_plot_path": str(spectrum_plot_path) if spectrum_plot_path else "",
            "spectrum_metadata_path": str(spectrum_metadata_path) if spectrum_metadata_path else "",
            "copied_plot_path": copied_plot,
            **_compact_dict(
                row,
                [
                    "spectral_pixels",
                    "wavelength_min",
                    "wavelength_max",
                    "flux_min",
                    "flux_median",
                    "flux_max",
                ],
            ),
        }
        photometric = _compact_dict(
            row,
            [
                "gaia_phot_g_mean_mag",
                "gaia_phot_bp_mean_mag",
                "gaia_phot_rp_mean_mag",
                "gaia_bp_rp",
                "absolute_g_mag",
                "parallax_quality_flag",
                "parallax_over_error",
                "parallax_fractional_error",
            ],
        )
        physical = _compact_dict(
            row,
            [
                "gaia_teff_gspphot",
                "gaia_logg_gspphot",
                "gaia_mh_gspphot",
                "gaia_parallax",
                "gaia_parallax_error",
                "gaia_pmra",
                "gaia_pmra_error",
                "gaia_pmdec",
                "gaia_pmdec_error",
                "gaia_pm_total_mas_yr",
                "gaia_ruwe",
                "gaia_visibility_periods_used",
                "gaia_astrometric_excess_noise",
            ],
        )
        object_payload = {
            "star_id": star_id,
            "objID": row.get("objID"),
            "specObjID": row.get("specObjID"),
            "ra": _json_value(row.get("ra")),
            "dec": _json_value(row.get("dec")),
            "match_status": row.get("match_status"),
            "candidate_count": _json_value(row.get("candidate_count")),
            "best_distance_arcsec": _json_value(row.get("best_distance_arcsec")),
            "gaia_source_id": _json_value(row.get("gaia_source_id")),
            "gaia_designation": _json_value(row.get("gaia_designation")),
            "paths": {
                "rgb_metadata": str(object_dir / "RGB" / "metadata.json"),
                "spectrogram_metadata": str(object_dir / "Spectrogram" / "metadata.json"),
                "photometric": str(object_dir / "Photometric" / "photometric.json"),
                "physical": str(object_dir / "Physical" / "physical.json"),
                "target": str(object_dir / "Target" / "target.json"),
            },
        }

        _write_json(object_dir / "RGB" / "metadata.json", rgb)
        _write_json(object_dir / "Spectrogram" / "metadata.json", spectrogram)
        _write_json(object_dir / "Photometric" / "photometric.json", photometric)
        _write_json(object_dir / "Physical" / "physical.json", physical)
        _write_json(object_dir / "Target" / "target.json", target)
        _write_json(object_dir / "object.json", object_payload)

        index_rows.append(
            {
                "star_id": star_id,
                "objID": row.get("objID"),
                "specObjID": row.get("specObjID"),
                "spectral_subclass_raw": target["spectral_subclass_raw"],
                "spectral_class_major": target["spectral_class_major"],
                "stellar_evolution_class": target["stellar_evolution_class"],
                "stellar_evolution_status": target["stellar_evolution_status"],
                "match_status": row.get("match_status"),
                "object_dir": str(object_dir),
            }
        )

    index = pd.DataFrame(index_rows)
    index.to_csv(output_dir / "star_objects_index.csv", index=False)
    return index
