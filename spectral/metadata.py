from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


IDENTIFIER_SETS = (
    ("obs_id",),
    ("ra", "dec"),
    ("plate", "mjd", "fiber"),
    ("plate", "mjd", "fiberid"),
)


TRACEABILITY_COLUMNS = [
    "image_path",
    "obs_id",
    "ra",
    "dec",
    "plate",
    "mjd",
    "fiber",
    "fiberid",
    "objid",
    "specobjid",
    "class",
    "subclass",
    "z",
    "z_err",
    "sn_median_all",
]


@dataclass
class MetadataLoadResult:
    records: pd.DataFrame
    rejected: pd.DataFrame
    summary: dict


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in df.columns:
        normalized = str(column).strip().lower()
        normalized = normalized.replace(" ", "_")
        normalized = normalized.replace("-", "_")
        if normalized == "fiberid":
            renamed[column] = "fiberid"
        elif normalized == "fiber_id":
            renamed[column] = "fiberid"
        elif normalized == "object_id":
            renamed[column] = "objid"
        elif normalized == "obj_id":
            renamed[column] = "objid"
        elif normalized == "spec_obj_id":
            renamed[column] = "specobjid"
        else:
            renamed[column] = normalized
    return df.rename(columns=renamed)


def _is_missing(value) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def _has_identifier(row: pd.Series) -> bool:
    for required in IDENTIFIER_SETS:
        if all(field in row.index and not _is_missing(row[field]) for field in required):
            return True
    return False


def _existing_path(path: Path) -> Optional[str]:
    if path.exists():
        return str(path)
    return None


def _resolve_from_parts(row: pd.Series, train_dir: str, test_dir: str) -> Optional[str]:
    if not {"split", "class_name", "image_filename"}.issubset(row.index):
        return None

    if _is_missing(row["split"]) or _is_missing(row["class_name"]) or _is_missing(row["image_filename"]):
        return None

    split = str(row["split"]).strip().lower()
    class_name = str(row["class_name"]).strip()
    image_filename = str(row["image_filename"]).strip()

    if split == "train":
        return _existing_path(Path(train_dir) / class_name / image_filename)
    if split == "test":
        return _existing_path(Path(test_dir) / class_name / image_filename)

    candidate = Path("data") / split / class_name / image_filename
    return _existing_path(candidate)


def _resolve_image_path(row: pd.Series, train_dir: str, test_dir: str) -> Optional[str]:
    if "image_path" in row.index and not _is_missing(row["image_path"]):
        path = Path(str(row["image_path"]).strip())
        if not path.is_absolute():
            path = Path.cwd() / path
        return _existing_path(path)

    return _resolve_from_parts(row, train_dir, test_dir)


def _is_star(row: pd.Series) -> bool:
    if "class_name" in row.index and not _is_missing(row["class_name"]):
        return str(row["class_name"]).strip().lower() == "star"

    if "class" in row.index and not _is_missing(row["class"]):
        return str(row["class"]).strip().lower() == "star"

    if "image_path" in row.index and not _is_missing(row["image_path"]):
        parts = [part.lower() for part in Path(str(row["image_path"])).parts]
        return "star" in parts

    return False


def _sample_records(df: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples is None or max_samples < 0 or len(df) <= max_samples:
        return df
    return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def _traceability_view(df: pd.DataFrame) -> pd.DataFrame:
    present = [column for column in TRACEABILITY_COLUMNS if column in df.columns]
    rest = [column for column in df.columns if column not in present]
    return df[present + rest]


def load_spectral_metadata(
    csv_path: str,
    train_dir: str,
    test_dir: str,
    max_samples: int = -1,
    seed: int = 42,
) -> MetadataLoadResult:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV nao encontrado: {csv_path}")

    df = _normalize_columns(pd.read_csv(path))

    accepted = []
    rejected = []

    for _, row in df.iterrows():
        reasons = []

        image_path = _resolve_image_path(row, train_dir, test_dir)
        if image_path is None:
            reasons.append("imagem sem caminho existente")

        if not _is_star(row):
            reasons.append("registro nao pertence a classe star")

        if not _has_identifier(row):
            reasons.append("sem obs_id, ra/dec ou plate/mjd/fiber")

        row_dict = row.to_dict()
        row_dict["image_path"] = image_path or row_dict.get("image_path", "")

        if reasons:
            row_dict["reject_reason"] = "; ".join(reasons)
            rejected.append(row_dict)
            continue

        accepted.append(row_dict)

    records = pd.DataFrame(accepted)
    rejected_df = pd.DataFrame(rejected)

    if records.empty:
        raise ValueError(
            "Nenhum registro espectral valido. O CSV precisa conter estrelas "
            "com image_path existente e obs_id, ra/dec ou plate/mjd/fiber."
        )

    records = _sample_records(records, max_samples=max_samples, seed=seed)
    records = _traceability_view(records).reset_index(drop=True)

    summary = {
        "metadata_csv": str(path),
        "total_rows": int(len(df)),
        "accepted_rows": int(len(records)),
        "rejected_rows": int(len(rejected_df)),
        "max_samples": int(max_samples),
    }

    return MetadataLoadResult(records=records, rejected=rejected_df, summary=summary)
