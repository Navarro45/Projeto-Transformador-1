from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class SDSSResolution:
    image_path: str
    obs_id: Optional[str]
    ra: Optional[float]
    dec: Optional[float]
    plate: Optional[int]
    mjd: Optional[int]
    fiber: Optional[int]
    metadata: dict


def _value(row: pd.Series, key: str):
    if key not in row.index:
        return None
    value = row[key]
    if pd.isna(value) or str(value).strip() == "":
        return None
    return value


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _as_int(value) -> Optional[int]:
    if value is None:
        return None
    return int(float(value))


class SDSSSpectrumResolver:
    """Resolve metadata rows to stable SDSS identifiers.

    This class never derives astronomical identifiers from filenames.
    """

    def resolve(self, row: pd.Series) -> SDSSResolution:
        image_path = _value(row, "image_path")
        if image_path is None or not Path(str(image_path)).exists():
            raise ValueError("imagem sem caminho existente")

        obs_id = _value(row, "obs_id")
        ra = _as_float(_value(row, "ra"))
        dec = _as_float(_value(row, "dec"))
        plate = _as_int(_value(row, "plate"))
        mjd = _as_int(_value(row, "mjd"))
        fiber = _as_int(_value(row, "fiber") or _value(row, "fiberid"))

        if obs_id is None and plate is not None and mjd is not None and fiber is not None:
            obs_id = f"{plate}-{mjd}-{fiber}"

        if obs_id is None and (ra is None or dec is None):
            raise ValueError("sem obs_id, ra/dec ou plate/mjd/fiber")

        return SDSSResolution(
            image_path=str(image_path),
            obs_id=str(obs_id) if obs_id is not None else None,
            ra=ra,
            dec=dec,
            plate=plate,
            mjd=mjd,
            fiber=fiber,
            metadata=row.to_dict(),
        )
