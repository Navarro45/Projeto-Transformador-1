from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from spectral.sdss_resolver import SDSSResolution


@dataclass
class SpectrumFetchResult:
    wavelength: np.ndarray
    flux: np.ndarray
    metadata: dict
    cache_hit: bool
    fits_path: str


class SpectrumGenerator:
    """Fetch real SDSS/SEGUE spectra through MAST and cache FITS files."""

    def __init__(
        self,
        cache_dir: str = "data/spectral_cache",
        cache_enabled: bool = True,
        search_radius_arcsec: float = 2.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_enabled = cache_enabled
        self.search_radius_arcsec = search_radius_arcsec
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, resolution: SDSSResolution) -> SpectrumFetchResult:
        cache_key = self._cache_key(resolution)
        fits_path = self.cache_dir / f"{cache_key}.fits"
        meta_path = self.cache_dir / f"{cache_key}.json"

        cache_hit = self.cache_enabled and fits_path.exists()

        if not cache_hit:
            products = self._query_products(resolution)
            self._download_product(products, fits_path)
            meta_path.write_text(
                json.dumps(
                    {
                        "resolution": asdict(resolution),
                        "cache_key": cache_key,
                        "search_radius_arcsec": self.search_radius_arcsec,
                    },
                    indent=4,
                ),
                encoding="utf-8",
            )

        wavelength, flux, fits_metadata = self._read_fits(fits_path)
        metadata = dict(resolution.metadata)
        metadata.update(fits_metadata)
        metadata["cache_key"] = cache_key
        metadata["fits_path"] = str(fits_path)

        return SpectrumFetchResult(
            wavelength=wavelength,
            flux=flux,
            metadata=metadata,
            cache_hit=cache_hit,
            fits_path=str(fits_path),
        )

    def _cache_key(self, resolution: SDSSResolution) -> str:
        if resolution.obs_id:
            stable = resolution.obs_id
        elif resolution.plate is not None and resolution.mjd is not None and resolution.fiber is not None:
            stable = f"{resolution.plate}-{resolution.mjd}-{resolution.fiber}"
        else:
            stable = f"{resolution.ra:.8f}_{resolution.dec:.8f}"
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]

    def _query_products(self, resolution: SDSSResolution):
        try:
            from astropy import units as u
            from astroquery.mast import Observations
        except ImportError as error:
            raise ImportError(
                "astroquery e astropy sao necessarios para baixar espectros SDSS reais."
            ) from error

        if resolution.obs_id:
            observations = Observations.query_criteria(
                provenance_name="SEGUE",
                obs_id=resolution.obs_id,
            )
        else:
            from astropy.coordinates import SkyCoord

            coordinates = SkyCoord(
                ra=resolution.ra * u.deg,
                dec=resolution.dec * u.deg,
            )
            observations = Observations.query_region(
                coordinates,
                radius=self.search_radius_arcsec * u.arcsec,
            )
            observations = self._filter_segue_observations(observations)

        if len(observations) == 0:
            raise ValueError("sem match SDSS/SEGUE no MAST")

        products = Observations.get_product_list(observations)
        if len(products) == 0:
            raise ValueError("match SDSS/SEGUE sem produtos FITS")

        return products

    def _filter_segue_observations(self, observations):
        if len(observations) == 0:
            return observations

        for column in ["provenance_name", "obs_collection"]:
            if column in observations.colnames:
                mask = [
                    "SEGUE" in str(value).upper() or "SDSS" in str(value).upper()
                    for value in observations[column]
                ]
                if any(mask):
                    return observations[mask]

        return observations

    def _download_product(self, products, fits_path: Path) -> None:
        try:
            from astroquery.mast import Observations
        except ImportError as error:
            raise ImportError("astroquery e necessario para baixar produtos MAST.") from error

        selected = products
        if "productFilename" in products.colnames:
            fits_mask = [
                str(name).lower().endswith((".fits", ".fit", ".fits.gz"))
                for name in products["productFilename"]
            ]
            if any(fits_mask):
                selected = products[fits_mask]

        manifest = Observations.download_products(
            selected[:1],
            download_dir=str(self.cache_dir),
        )

        if len(manifest) == 0 or "Local Path" not in manifest.colnames:
            raise ValueError("download MAST nao retornou arquivo local")

        downloaded = Path(str(manifest[0]["Local Path"]))
        if not downloaded.exists():
            raise ValueError(f"produto MAST baixado nao encontrado: {downloaded}")

        fits_path.write_bytes(downloaded.read_bytes())

    def _read_fits(self, fits_path: Path):
        try:
            from astropy.io import fits
        except ImportError as error:
            raise ImportError("astropy e necessario para ler FITS SDSS.") from error

        with fits.open(fits_path) as hdul:
            hdu = self._find_coadd_hdu(hdul)
            data = hdu.data
            names = {name.lower(): name for name in data.names}

            if "loglam" not in names or "flux" not in names:
                raise ValueError("FITS invalido: extensao COADD sem loglam/flux")

            loglam = np.asarray(data[names["loglam"]], dtype=np.float64)
            flux = np.asarray(data[names["flux"]], dtype=np.float64)
            wavelength = np.power(10.0, loglam)

            metadata = self._extract_metadata(hdul, hdu)
            metadata["fits_hdu"] = hdu.name

            return wavelength, flux, metadata

    def _find_coadd_hdu(self, hdul):
        for hdu in hdul:
            if hdu.name.upper() == "COADD" and getattr(hdu, "data", None) is not None:
                return hdu
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            names = getattr(data, "names", None)
            if names and {"loglam", "flux"}.issubset({name.lower() for name in names}):
                return hdu
        raise ValueError("FITS invalido: extensao COADD nao encontrada")

    def _extract_metadata(self, hdul, coadd_hdu):
        fields = [
            "CLASS",
            "SUBCLASS",
            "PLATE",
            "MJD",
            "FIBERID",
            "OBJID",
            "Z",
            "Z_ERR",
            "SN_MEDIAN_ALL",
        ]

        metadata = {}

        for key in fields:
            value = coadd_hdu.header.get(key)
            if value is not None:
                metadata[key.lower()] = self._json_safe(value)

        for hdu in hdul:
            data = getattr(hdu, "data", None)
            names = getattr(data, "names", None)
            if not names or len(data) == 0:
                continue

            lower_to_original = {name.lower(): name for name in names}
            for key in fields:
                lower = key.lower()
                if lower in metadata or lower not in lower_to_original:
                    continue
                value = data[lower_to_original[lower]][0]
                metadata[lower] = self._json_safe(value)

        return metadata

    def _json_safe(self, value):
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore").strip()
        return value
