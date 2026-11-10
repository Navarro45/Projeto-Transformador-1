"""Cria o manifesto unico da hierarquia usando apenas artefatos locais."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.spectral_labels import normalize_spectral_label  # noqa: E402
from shared.stellar_hierarchy import (  # noqa: E402
    SPECIAL_STELLAR_CLASSES,
    UNRESOLVED_LABEL,
    major_spectral_class,
    roman_luminosity_from_gaia,
)


CLASS_BY_FOLDER = {"galaxy": "GALAXY", "quasar": "QSO", "star": "STAR"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gera um manifesto sem objetos repetidos para os tres niveis. "
            "Nenhuma imagem e baixada, copiada ou reprocessada."
        )
    )
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument(
        "--metadata-csv",
        default="dataset/metadata/spectra_metadata.csv",
        help="Metadados SDSS do conjunto original.",
    )
    parser.add_argument(
        "--gaia-verified-csv",
        default="dataset/gaia_verified_sdss/verified_gaia_sdss_stars.csv",
        help="Estrelas adicionais ja verificadas e baixadas localmente.",
    )
    parser.add_argument(
        "--output",
        default="dataset/hierarchy/roman_spectral_manifest.csv",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _ROOT / path


def build_manifest(
    dataset_root: Path,
    metadata_csv: Path,
    gaia_verified_csv: Path,
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_csv, dtype=str).fillna("")
    required = {"objID", "specObjID", "class", "subclass"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError("Metadados SDSS sem colunas: " + ", ".join(missing))
    if metadata["objID"].duplicated().any():
        raise ValueError("O metadata SDSS possui objID repetido")
    base_by_id = metadata.set_index("objID", drop=False).to_dict("index")

    gaia_by_id: dict[str, dict] = {}
    if gaia_verified_csv.is_file():
        gaia = pd.read_csv(gaia_verified_csv, dtype=str).fillna("")
        if gaia["objID"].duplicated().any():
            raise ValueError("O conjunto Gaia verificado possui objID repetido")
        overlap = set(base_by_id).intersection(gaia["objID"])
        if overlap:
            raise ValueError(f"Objetos repetidos entre SDSS e Gaia verificado: {len(overlap)}")
        gaia_by_id = gaia.set_index("objID", drop=False).to_dict("index")

    rows: list[dict] = []
    observed_ids: set[str] = set()
    for split in ("train", "val", "test"):
        for folder, object_class in CLASS_BY_FOLDER.items():
            split_dir = dataset_root / split / folder
            if not split_dir.is_dir():
                raise FileNotFoundError(f"Split de imagens ausente: {split_dir}")
            for image_path in sorted(split_dir.iterdir()):
                if not image_path.is_file():
                    continue
                object_id = image_path.stem
                if object_id in observed_ids:
                    raise ValueError(f"Objeto presente em mais de um split: {object_id}")
                observed_ids.add(object_id)

                source = "sdss_original"
                if object_id in base_by_id:
                    record = base_by_id[object_id]
                    spec_object_id = str(record["specObjID"])
                    raw_subclass = str(record["subclass"]) if object_class == "STAR" else ""
                elif object_id in gaia_by_id and object_class == "STAR":
                    source = "gaia_verified_sdss"
                    record = gaia_by_id[object_id]
                    spec_object_id = str(record["specObjID"])
                    raw_subclass = _read_local_subclass(dataset_root, spec_object_id)
                else:
                    raise ValueError(
                        f"Nao foi possivel associar metadados ao objeto {object_id} ({image_path})"
                    )

                labels = (
                    normalize_spectral_label(raw_subclass)
                    if object_class == "STAR"
                    else {
                        "spectral_subclass_raw": "",
                        "spectral_parent_label": "",
                        "spectral_leaf_label": "",
                    }
                )
                hierarchy_labels = (
                    _stellar_hierarchy_labels(labels, gaia_by_id.get(object_id))
                    if object_class == "STAR"
                    else _empty_hierarchy_labels()
                )
                rows.append(
                    {
                        "object_id": object_id,
                        "spec_object_id": spec_object_id,
                        "split": split,
                        "image_path": str(image_path.resolve()),
                        "object_class": object_class,
                        **labels,
                        **hierarchy_labels,
                        "source": source,
                    }
                )

    manifest = pd.DataFrame(rows)
    if manifest["object_id"].duplicated().any():
        raise ValueError("O manifesto resultante possui object_id repetido")
    nonempty_spec = manifest.loc[manifest["spec_object_id"].ne(""), "spec_object_id"]
    if nonempty_spec.duplicated().any():
        duplicates = nonempty_spec[nonempty_spec.duplicated()].head(5).tolist()
        raise ValueError("spec_object_id repetido no manifesto: " + ", ".join(duplicates))
    return manifest.sort_values(["split", "object_class", "object_id"]).reset_index(drop=True)


def _empty_hierarchy_labels() -> dict:
    return {
        "stellar_level2_label": "",
        "stellar_level2_raw": "",
        "stellar_level2_source": "",
        "stellar_level2_method": "",
        "stellar_level2_confidence": "",
        "stellar_level3_label": "",
        "stellar_level3_source": "",
        "hierarchy_supported": "",
        "hierarchy_stop_reason": "",
    }


def _stellar_hierarchy_labels(spectral_labels: dict, gaia_record: dict | None) -> dict:
    spectral_parent = str(spectral_labels.get("spectral_parent_label") or "")
    if spectral_parent in SPECIAL_STELLAR_CLASSES:
        return {
            "stellar_level2_label": spectral_parent,
            "stellar_level2_raw": str(spectral_labels.get("spectral_subclass_raw") or ""),
            "stellar_level2_source": "sdss_subclass",
            "stellar_level2_method": "special_class_override",
            "stellar_level2_confidence": "catalogued",
            "stellar_level3_label": "",
            "stellar_level3_source": "",
            "hierarchy_supported": "true",
            "hierarchy_stop_reason": "special_level2_leaf",
        }

    gaia_record = gaia_record or {}
    raw_luminosity = str(gaia_record.get("gaia_luminosity_class") or "").strip()
    roman = roman_luminosity_from_gaia(raw_luminosity)
    raw_spectral = str(
        gaia_record.get("gaia_spectral_class")
        or gaia_record.get("spectraltype_esphs")
        or ""
    ).strip()
    spectral = major_spectral_class(raw_spectral)
    if not roman:
        return {
            "stellar_level2_label": UNRESOLVED_LABEL,
            "stellar_level2_raw": raw_luminosity,
            "stellar_level2_source": "",
            "stellar_level2_method": "",
            "stellar_level2_confidence": "",
            "stellar_level3_label": spectral,
            "stellar_level3_source": "gaia_spectraltype_esphs" if spectral else "",
            "hierarchy_supported": "false",
            "hierarchy_stop_reason": "missing_luminosity_class",
        }
    return {
        "stellar_level2_label": roman,
        "stellar_level2_raw": raw_luminosity,
        "stellar_level2_source": "gaia_evolstage_flame",
        "stellar_level2_method": "gaia_evolution_to_roman",
        "stellar_level2_confidence": "derived",
        "stellar_level3_label": spectral,
        "stellar_level3_source": "gaia_spectraltype_esphs" if spectral else "",
        "hierarchy_supported": "true" if spectral else "false",
        "hierarchy_stop_reason": "" if spectral else "missing_spectral_class",
    }


def _read_local_subclass(dataset_root: Path, spec_object_id: str) -> str:
    path = dataset_root / "metadata" / "spectra" / "star" / f"{spec_object_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Metadata espectral local ausente: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    subclass = str(payload.get("subclass") or "").strip()
    if not subclass:
        raise ValueError(f"Subclass SDSS ausente em {path}")
    return subclass


def build_summary(manifest: pd.DataFrame) -> dict:
    stars = manifest.loc[manifest["object_class"].eq("STAR")]
    variants = (
        stars.groupby(["spectral_parent_label", "spectral_leaf_label"])
        .size()
        .reset_index(name="count")
    )
    return {
        "total_objects": int(len(manifest)),
        "unique_object_ids": int(manifest["object_id"].nunique()),
        "unique_spec_object_ids": int(manifest["spec_object_id"].nunique()),
        "split_counts": {
            str(key): int(value) for key, value in manifest["split"].value_counts().items()
        },
        "object_class_counts": {
            str(key): int(value)
            for key, value in manifest["object_class"].value_counts().items()
        },
        "spectral_parent_counts": {
            str(key): int(value)
            for key, value in stars["spectral_parent_label"].value_counts().items()
        },
        "stellar_level2_counts": {
            str(key): int(value)
            for key, value in stars["stellar_level2_label"].value_counts().items()
        },
        "stellar_level3_counts": {
            str(key): int(value)
            for key, value in stars["stellar_level3_label"].value_counts().items()
            if str(key)
        },
        "variant_count_by_parent": {
            str(parent): int(group["spectral_leaf_label"].nunique())
            for parent, group in variants.groupby("spectral_parent_label")
        },
        "source_counts": {
            str(key): int(value) for key, value in manifest["source"].value_counts().items()
        },
    }


def main() -> int:
    args = parse_args()
    output = resolve_path(args.output)
    manifest = build_manifest(
        dataset_root=resolve_path(args.dataset_root),
        metadata_csv=resolve_path(args.metadata_csv),
        gaia_verified_csv=resolve_path(args.gaia_verified_csv),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    summary_path = output.with_name(f"{output.stem}_summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(build_summary(manifest), handle, indent=2)
    print(f"Manifesto hierarquico: {output}")
    print(f"Resumo: {summary_path}")
    print(f"Objetos unicos: {len(manifest)}")
    print("Nenhuma imagem foi baixada, copiada ou reprocessada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
