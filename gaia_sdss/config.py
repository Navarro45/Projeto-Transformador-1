from __future__ import annotations

from dataclasses import dataclass


GAIA_MATCH_RADIUS_ARCSEC = 1.0
GAIA_AMBIGUITY_DELTA_ARCSEC = 0.2
GAIA_HIGH_PM_MAS_YR = 50.0

GAIA_COLUMNS = [
    "source_id",
    "designation",
    "ra",
    "dec",
    "ref_epoch",
    "parallax",
    "parallax_error",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "bp_rp",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "ruwe",
    "visibility_periods_used",
    "astrometric_excess_noise",
    "classprob_dsc_combmod_star",
    "classprob_dsc_combmod_quasar",
    "classprob_dsc_combmod_galaxy",
]

TEFF_SPECTRAL_LIMITS = [
    ("O", 30000.0, float("inf")),
    ("B", 10000.0, 30000.0),
    ("A", 7500.0, 10000.0),
    ("F", 6000.0, 7500.0),
    ("G", 5200.0, 6000.0),
    ("K", 3700.0, 5200.0),
    ("M", 0.0, 3700.0),
]


@dataclass(frozen=True)
class GaiaIntegrationConfig:
    match_radius_arcsec: float = GAIA_MATCH_RADIUS_ARCSEC
    ambiguity_delta_arcsec: float = GAIA_AMBIGUITY_DELTA_ARCSEC
    high_pm_mas_yr: float = GAIA_HIGH_PM_MAS_YR
    batch_size: int = 500
