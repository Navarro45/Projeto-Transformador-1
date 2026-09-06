from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np


@dataclass
class SpectralPreprocessorConfig:
    spectral_length: int = 1024
    wavelength_min: float = 3800.0
    wavelength_max: float = 9200.0
    min_valid_fraction: float = 0.70
    max_nan_fraction: float = 0.20
    normalization: str = "robust"


class SpectralPreprocessor:
    def __init__(self, config: SpectralPreprocessorConfig):
        self.config = config
        self.grid = np.linspace(
            config.wavelength_min,
            config.wavelength_max,
            config.spectral_length,
            dtype=np.float64,
        )

    def transform(self, wavelength, flux) -> Tuple[np.ndarray, dict]:
        wavelength = np.asarray(wavelength, dtype=np.float64)
        flux = np.asarray(flux, dtype=np.float64)

        if wavelength.shape != flux.shape:
            raise ValueError("wavelength e flux possuem tamanhos diferentes")

        total = len(flux)
        if total == 0:
            raise ValueError("espectro vazio")

        finite = np.isfinite(wavelength) & np.isfinite(flux)
        nan_fraction = 1.0 - (float(np.count_nonzero(finite)) / float(total))
        if nan_fraction > self.config.max_nan_fraction:
            raise ValueError("NaN/infinito excessivo")

        wavelength = wavelength[finite]
        flux = flux[finite]

        valid_range = (
            (wavelength >= self.config.wavelength_min) &
            (wavelength <= self.config.wavelength_max)
        )
        wavelength = wavelength[valid_range]
        flux = flux[valid_range]

        if len(flux) == 0:
            raise ValueError("cobertura espectral insuficiente")

        coverage = (
            wavelength.max() - wavelength.min()
        ) / (
            self.config.wavelength_max - self.config.wavelength_min
        )
        if coverage < self.config.min_valid_fraction:
            raise ValueError("cobertura espectral insuficiente")

        order = np.argsort(wavelength)
        wavelength = wavelength[order]
        flux = flux[order]

        unique_wavelength, unique_indices = np.unique(wavelength, return_index=True)
        flux = flux[unique_indices]
        wavelength = unique_wavelength

        if np.allclose(flux, 0):
            raise ValueError("fluxo totalmente nulo")

        interpolated = np.interp(self.grid, wavelength, flux)
        normalized = self._normalize(interpolated)

        quality = {
            "input_pixels": int(total),
            "valid_pixels": int(len(flux)),
            "nan_fraction": float(nan_fraction),
            "coverage_fraction": float(coverage),
            "wavelength_min": float(wavelength.min()),
            "wavelength_max": float(wavelength.max()),
        }

        return normalized.astype(np.float32), quality

    def _normalize(self, flux: np.ndarray) -> np.ndarray:
        if self.config.normalization == "none":
            return flux

        if self.config.normalization == "standard":
            mean = np.mean(flux)
            std = np.std(flux)
            if std <= 0:
                raise ValueError("fluxo sem variancia")
            return (flux - mean) / std

        median = np.median(flux)
        q75, q25 = np.percentile(flux, [75, 25])
        scale = q75 - q25
        if scale <= 0:
            scale = np.std(flux)
        if scale <= 0:
            raise ValueError("fluxo sem variancia")
        return (flux - median) / scale

    def to_dict(self) -> dict:
        payload = asdict(self.config)
        payload["grid"] = self.grid.tolist()
        return payload
