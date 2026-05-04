import hashlib
import importlib
import os
import numpy as np
from PIL import Image


class SpectrumGenerator:
    """
    Adapter for pluggable image->spectrum generation backends.
    - heuristic: deterministic fallback using image intensity profile.
    - mast_sdss: fetches SDSS spectra from MAST/SEGUE products.
    - external: dynamically imports callable from config string:
      "package.module:function_name"
    """

    def __init__(self, config):
        self.length = config.SPECTRAL_LENGTH
        self.backend = getattr(config, "SPECTRUM_GENERATOR_BACKEND", "heuristic")
        self.callable_path = getattr(config, "SPECTRUM_GENERATOR_CALLABLE", "")
        self.cache_enabled = getattr(config, "SPECTRUM_CACHE_ENABLED", True)
        self.allow_heuristic_fallback = getattr(config, "SPECTRUM_ALLOW_HEURISTIC_FALLBACK", True)
        self.mast_provenance = getattr(config, "MAST_SDSS_PROVENANCE", "SEGUE")
        self.cache_dir = config.PATHS.get("generated_spectra", "")
        self.mast_products_dir = os.path.join(self.cache_dir, "mast_products")
        self._external_callable = self._load_external_callable()

        if self.cache_enabled and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(self.mast_products_dir, exist_ok=True)

    def _load_external_callable(self):
        if self.backend != "external" or not self.callable_path:
            return None
        try:
            module_name, fn_name = self.callable_path.split(":")
            module = importlib.import_module(module_name)
            return getattr(module, fn_name)
        except Exception as exc:
            print(f"⚠️ Falha ao carregar gerador externo ({self.callable_path}): {exc}")
            print("➡️ Usando fallback heurístico para espectro.")
            self.backend = "heuristic"
            return None

    def _cache_path(self, image_path):
        key = hashlib.md5(image_path.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.npy")

    def _cache_path_with_metadata(self, image_path, metadata):
        obs_id = (metadata or {}).get("obs_id", "")
        if not obs_id:
            return self._cache_path(image_path)
        key_source = f"{image_path}|{obs_id}"
        key = hashlib.md5(key_source.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.npy")

    def _normalize_length(self, spectrum):
        spectrum = np.asarray(spectrum, dtype=np.float32).flatten()
        if spectrum.size == 0:
            spectrum = np.zeros(self.length, dtype=np.float32)
        if spectrum.size < self.length:
            spectrum = np.pad(spectrum, (0, self.length - spectrum.size))
        else:
            spectrum = spectrum[:self.length]
        spectrum = (spectrum - np.mean(spectrum)) / (np.std(spectrum) + 1e-8)
        return spectrum.astype(np.float32)

    def _heuristic_spectrum(self, image):
        gray = image.convert("L")
        arr = np.asarray(gray, dtype=np.float32) / 255.0
        profile = arr.mean(axis=0)
        x_old = np.linspace(0.0, 1.0, num=profile.shape[0], dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, num=self.length, dtype=np.float32)
        spectrum = np.interp(x_new, x_old, profile)
        return self._normalize_length(spectrum)

    def _spectrum_from_mast(self, metadata):
        obs_id = (metadata or {}).get("obs_id", "")
        if not obs_id:
            return None
        try:
            from astroquery.mast import Observations
            from astropy.io import fits
        except Exception as exc:
            print(f"⚠️ Dependências MAST não disponíveis ({exc}).")
            return None

        try:
            obs = Observations.query_criteria(
                provenance_name=self.mast_provenance,
                obs_id=obs_id
            )
            if len(obs) == 0:
                return None

            products = Observations.get_product_list(obs[0])
            manifest = Observations.download_products(
                products,
                flat=True,
                mrp_only=True,
                download_dir=self.mast_products_dir,
                cache=True
            )
            if len(manifest) == 0:
                return None

            fits_path = manifest["Local Path"][0]
            with fits.open(fits_path, memmap=False) as hdul:
                flux = hdul["COADD"].data["FLUX"]
            return self._normalize_length(flux)
        except Exception as exc:
            print(f"⚠️ Falha ao buscar espectro MAST para {obs_id}: {exc}")
            return None

    def generate(self, image_path, metadata=None):
        cache_path = self._cache_path_with_metadata(image_path, metadata)
        if self.cache_enabled and self.cache_dir:
            if os.path.exists(cache_path):
                return np.load(cache_path)

        image = Image.open(image_path).convert("RGB")
        if self.backend == "mast_sdss":
            spectrum = self._spectrum_from_mast(metadata)
            if spectrum is None and self.allow_heuristic_fallback:
                spectrum = self._heuristic_spectrum(image)
        elif self.backend == "external" and self._external_callable is not None:
            spectrum = self._external_callable(image)
            spectrum = self._normalize_length(spectrum)
        else:
            spectrum = self._heuristic_spectrum(image)

        if self.cache_enabled and self.cache_dir:
            np.save(cache_path, spectrum)
        return spectrum
