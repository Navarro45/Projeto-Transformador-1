from __future__ import annotations

from pathlib import Path

import pandas as pd


def _normalize_key(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _existing_or_none(value) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return ""
    path = Path(str(value))
    return str(path) if path.exists() else str(path)


def load_local_sdss_dataset(input_path: str | Path, manifest_path: str | Path | None = None) -> pd.DataFrame:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset local nao encontrado: {input_path}")

    df = pd.read_csv(input_path)
    df.columns = [str(column).strip() for column in df.columns]

    if "specObjID" not in df.columns:
        raise ValueError("Dataset local precisa conter a coluna specObjID.")

    df["specObjID"] = df["specObjID"].map(_normalize_key)
    if "objID" in df.columns:
        df["objID"] = df["objID"].map(_normalize_key)

    if "subclass" in df.columns and "pipeline_subclass" not in df.columns:
        df["pipeline_subclass"] = df["subclass"].fillna("SEM_SUBCLASSE")

    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        if manifest_path.exists():
            manifest = pd.read_csv(
                manifest_path,
                usecols=lambda column: column in {
                    "objID",
                    "specObjID",
                    "ra",
                    "dec",
                    "redshift",
                    "image_path",
                    "spectrum_path",
                    "spectrum_metadata_path",
                },
            )
            manifest["specObjID"] = manifest["specObjID"].map(_normalize_key)
            if "objID" in manifest.columns:
                manifest["objID"] = manifest["objID"].map(_normalize_key)

            merge_columns = [
                column for column in manifest.columns
                if column == "specObjID" or column not in df.columns or column in {"ra", "dec"}
            ]
            df = df.merge(
                manifest[merge_columns].drop_duplicates("specObjID"),
                on="specObjID",
                how="left",
                suffixes=("", "_manifest"),
            )

            for column in ("ra", "dec"):
                manifest_column = f"{column}_manifest"
                if manifest_column in df.columns:
                    if column in df.columns:
                        df[column] = df[column].combine_first(df[manifest_column])
                    else:
                        df[column] = df[manifest_column]
                    df = df.drop(columns=[manifest_column])

    if "ra" not in df.columns or "dec" not in df.columns:
        raise ValueError("Nao encontrei RA/DEC no dataset local nem no manifesto.")

    df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
    df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
    df["has_radec"] = df["ra"].notna() & df["dec"].notna()

    for column in ("image_path", "spectrum_path", "spectrum_metadata_path"):
        if column in df.columns:
            df[column] = df[column].map(_existing_or_none)

    gaia_columns = [
        column for column in df.columns
        if "gaia" in column.lower() or column.lower() in {"source_id", "designation"}
    ]
    df.attrs["existing_gaia_columns"] = gaia_columns

    return df
