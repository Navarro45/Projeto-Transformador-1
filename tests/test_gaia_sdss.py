from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from gaia_sdss import gaia_client
from gaia_sdss.local_data import load_local_sdss_dataset
from gaia_sdss.object_dataset import build_star_object_dataset
from gaia_sdss.matching import (
    ERROR,
    MATCH_AMBIGUOUS,
    MATCH_BEST,
    MATCH_UNIQUE,
    NO_MATCH,
    angular_separation_arcsec,
    derive_spectral_class_from_teff,
    select_best_match,
)
from gaia_sdss.reporting import build_consolidated_dataset, generate_reports
from gaia_sdss.stellar import absolute_g_mag, derive_stellar_evolution_class, parse_sdss_subclass
from scripts.gaia_sdss_integration import filter_by_classes, parse_args


class GaiaSdssMatchingTests(unittest.TestCase):
    def test_angular_separation_arcsec(self):
        self.assertAlmostEqual(angular_separation_arcsec(0, 0, 0, 1), 3600.0, places=6)

    def test_select_unique_match(self):
        result = select_best_match([{"source_id": 1, "separation_arcsec": 0.12}], 1.0, 0.2)
        self.assertEqual(result["match_status"], MATCH_UNIQUE)
        self.assertEqual(result["best"]["source_id"], 1)

    def test_select_best_match(self):
        result = select_best_match(
            [
                {"source_id": 2, "separation_arcsec": 0.7},
                {"source_id": 1, "separation_arcsec": 0.1},
            ],
            1.0,
            0.2,
        )
        self.assertEqual(result["match_status"], MATCH_BEST)
        self.assertEqual(result["best"]["source_id"], 1)

    def test_select_ambiguous_match(self):
        result = select_best_match(
            [
                {"source_id": 1, "separation_arcsec": 0.10},
                {"source_id": 2, "separation_arcsec": 0.25},
            ],
            1.0,
            0.2,
        )
        self.assertEqual(result["match_status"], MATCH_AMBIGUOUS)
        self.assertTrue(result["is_ambiguous"])

    def test_no_match(self):
        result = select_best_match([], 1.0, 0.2)
        self.assertEqual(result["match_status"], NO_MATCH)

    def test_derive_spectral_class_from_teff(self):
        self.assertEqual(derive_spectral_class_from_teff(5772)["gaia_derived_class"], "G")
        self.assertEqual(derive_spectral_class_from_teff(None)["gaia_derived_status"], "MISSING")
        self.assertEqual(derive_spectral_class_from_teff(-1)["gaia_derived_status"], "INVALID")


class GaiaSdssStellarTests(unittest.TestCase):
    def test_parse_sdss_subclass(self):
        cases = {
            "M3": ("M", ""),
            "F5": ("F", ""),
            "A0": ("A", ""),
            "A1V": ("A", "V"),
            "A1III": ("A", "III"),
            "A0IVn (25642)": ("A", "IV"),
            "WD": ("WD", ""),
            "Carbon_lines": ("CARBON", ""),
        }
        for raw, expected in cases.items():
            parsed = parse_sdss_subclass(raw)
            self.assertEqual((parsed["spectral_class_major"], parsed["roman_luminosity_class_raw"]), expected)

    def test_absolute_g_mag(self):
        self.assertAlmostEqual(absolute_g_mag(10.0, 10.0), 5.0, places=6)
        self.assertIsNone(absolute_g_mag(10.0, -1.0))

    def test_derive_white_dwarf_from_subclass(self):
        result = derive_stellar_evolution_class({"pipeline_subclass": "WD"})
        self.assertEqual(result["stellar_evolution_class"], "WHITE_DWARF")
        self.assertEqual(result["stellar_evolution_status"], "DERIVED")

    def test_derive_main_sequence_dwarf(self):
        result = derive_stellar_evolution_class(
            {
                "pipeline_subclass": "G2",
                "gaia_phot_g_mean_mag": 10.0,
                "gaia_parallax": 10.0,
                "gaia_parallax_error": 0.5,
                "gaia_bp_rp": 0.8,
            }
        )
        self.assertEqual(result["stellar_evolution_class"], "MAIN_SEQUENCE_DWARF")
        self.assertEqual(result["parallax_quality_flag"], "GOOD")

    def test_derive_red_giant(self):
        result = derive_stellar_evolution_class(
            {
                "pipeline_subclass": "K3",
                "gaia_logg_gspphot": 2.4,
                "gaia_bp_rp": 1.4,
            }
        )
        self.assertEqual(result["stellar_evolution_class"], "RED_GIANT")

    def test_derive_giant_and_subgiant(self):
        giant = derive_stellar_evolution_class({"pipeline_subclass": "A0", "gaia_logg_gspphot": 2.7, "gaia_bp_rp": 0.2})
        subgiant = derive_stellar_evolution_class({"pipeline_subclass": "F5", "gaia_logg_gspphot": 3.8})
        self.assertEqual(giant["stellar_evolution_class"], "GIANT")
        self.assertEqual(subgiant["stellar_evolution_class"], "SUBGIANT")

    def test_missing_and_invalid_gaia_data(self):
        missing = derive_stellar_evolution_class({"pipeline_subclass": "M3"})
        invalid = derive_stellar_evolution_class({"pipeline_subclass": "M3", "gaia_parallax": -1.0})
        self.assertEqual(missing["stellar_evolution_class"], "UNKNOWN")
        self.assertEqual(missing["stellar_evolution_status"], "MISSING_GAIA_PARALLAX")
        self.assertEqual(invalid["stellar_evolution_status"], "INVALID_GAIA_PARALLAX")


class GaiaSdssDataTests(unittest.TestCase):
    def test_cli_defaults_to_star_and_small_batches(self):
        with patch("sys.argv", ["gaia_sdss_integration.py"]):
            args = parse_args()
        self.assertEqual(args.classes, ["STAR"])
        self.assertEqual(args.batch_size, 50)

    def test_filter_by_classes_defaults_behavior_helper(self):
        df = pd.DataFrame(
            [
                {"specObjID": "s1", "spectralClass": "STAR"},
                {"specObjID": "s2", "spectralClass": "GALAXY"},
                {"specObjID": "s3", "spectralClass": "QSO"},
            ]
        )
        filtered = filter_by_classes(df, ["STAR"])
        self.assertEqual(filtered["specObjID"].tolist(), ["s1"])

        filtered = filter_by_classes(df, ["GALAXY", "QSO", "STAR"])
        self.assertEqual(set(filtered["specObjID"]), {"s1", "s2", "s3"})

    def test_load_local_dataset_uses_manifest_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / "spectra_metadata.csv"
            manifest = root / "download_manifest.csv"
            pd.DataFrame(
                [
                    {
                        "objID": "o1",
                        "specObjID": "s1",
                        "spectralClass": "STAR",
                        "subclass": "G2",
                    }
                ]
            ).to_csv(local, index=False)
            pd.DataFrame(
                [
                    {
                        "objID": "o1",
                        "specObjID": "s1",
                        "ra": 10.0,
                        "dec": -2.0,
                    }
                ]
            ).to_csv(manifest, index=False)

            df = load_local_sdss_dataset(local, manifest)
            self.assertTrue(bool(df.loc[0, "has_radec"]))
            self.assertEqual(df.loc[0, "pipeline_subclass"], "G2")

    def test_build_consolidated_and_reports_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = pd.DataFrame(
                [
                    {
                        "objID": "o1",
                        "specObjID": "s1",
                        "spectralClass": "STAR",
                        "pipeline_subclass": "G2",
                        "ra": 10.0,
                        "dec": -2.0,
                        "has_radec": True,
                    }
                ]
            )
            gaia = pd.DataFrame(
                [
                    {
                        "objID": "o1",
                        "specObjID": "s1",
                        "match_status": MATCH_UNIQUE,
                        "candidate_count": 1,
                        "best_distance_arcsec": 0.1,
                        "source_id": 123,
                        "teff_gspphot": 5700.0,
                        "phot_g_mean_mag": 10.0,
                        "parallax": 10.0,
                        "parallax_error": 0.5,
                        "bp_rp": 0.8,
                        "pmra": 1.0,
                        "pmdec": 2.0,
                        "classprob_dsc_combmod_star": 0.9,
                    }
                ]
            )

            consolidated = build_consolidated_dataset(local, gaia)
            self.assertEqual(consolidated.loc[0, "gaia_derived_class"], "G")
            self.assertEqual(consolidated.loc[0, "spectral_subclass_raw"], "G2")
            self.assertEqual(consolidated.loc[0, "stellar_evolution_class"], "MAIN_SEQUENCE_DWARF")
            self.assertEqual(consolidated.loc[0, "spectral_agreement_status"], "AGREE")

            summary = generate_reports(consolidated, root, {"test": True})
            self.assertEqual(summary["total_local_objects"], 1)
            self.assertEqual(summary["stellar_evolution_class_counts"], {"MAIN_SEQUENCE_DWARF": 1})
            self.assertTrue((root / "reports" / "gaia_sdss_report.md").exists())
            self.assertTrue((root / "plots" / "match_status_counts.png").exists())
            self.assertTrue((root / "reports" / "stellar_evolution_class_counts.csv").exists())

    def test_cache_error_is_not_retried_without_retry_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.csv"
            pd.DataFrame(
                [
                    {"specObjID": "s1", "objID": "o1", "match_status": ERROR},
                ]
            ).to_csv(cache_path, index=False)
            rows = pd.DataFrame(
                [
                    {"specObjID": "s1", "objID": "o1", "ra": 1.0, "dec": 1.0, "has_radec": True},
                ]
            )

            with patch.object(gaia_client, "_query_individual_batch") as query:
                result = gaia_client.match_rows(
                    rows,
                    cache_path,
                    radius_arcsec=1.0,
                    ambiguity_delta_arcsec=0.2,
                    batch_size=1,
                    retry_errors=False,
                    use_batch=False,
                )

            query.assert_not_called()
            self.assertEqual(result.loc[0, "match_status"], ERROR)

    def test_retry_errors_requeries_only_error_rows_and_preserves_valid_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.csv"
            pd.DataFrame(
                [
                    {"specObjID": "s1", "objID": "o1", "match_status": MATCH_UNIQUE, "source_id": 10},
                    {"specObjID": "s2", "objID": "o2", "match_status": ERROR},
                    {"specObjID": "outside", "objID": "o3", "match_status": ERROR},
                ]
            ).to_csv(cache_path, index=False)
            rows = pd.DataFrame(
                [
                    {"specObjID": "s1", "objID": "o1", "ra": 1.0, "dec": 1.0, "has_radec": True},
                    {"specObjID": "s2", "objID": "o2", "ra": 2.0, "dec": 2.0, "has_radec": True},
                    {"specObjID": "s3", "objID": "o4", "ra": 3.0, "dec": 3.0, "has_radec": True},
                ]
            )

            def fake_query(batch, radius_arcsec):
                self.assertEqual(batch["specObjID"].tolist(), ["s2"])
                return pd.DataFrame(
                    [
                        {
                            "specObjID": "s2",
                            "objID": "o2",
                            "source_id": 20,
                            "separation_arcsec": 0.1,
                        }
                    ]
                )

            with patch.object(gaia_client, "_query_individual_batch", side_effect=fake_query) as query:
                result = gaia_client.match_rows(
                    rows,
                    cache_path,
                    radius_arcsec=1.0,
                    ambiguity_delta_arcsec=0.2,
                    batch_size=1,
                    retry_errors=True,
                    use_batch=False,
                )

            query.assert_called_once()
            by_id = result.set_index("specObjID")
            self.assertEqual(by_id.loc["s1", "match_status"], MATCH_UNIQUE)
            self.assertEqual(by_id.loc["s2", "match_status"], MATCH_UNIQUE)
            self.assertEqual(by_id.loc["outside", "match_status"], ERROR)
            self.assertNotIn("s3", by_id.index)

    def test_failed_batch_does_not_fallback_to_individual_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.csv"
            rows = pd.DataFrame(
                [
                    {"specObjID": "s1", "objID": "o1", "ra": 1.0, "dec": 1.0, "has_radec": True},
                    {"specObjID": "s2", "objID": "o2", "ra": 2.0, "dec": 2.0, "has_radec": True},
                ]
            )

            with (
                patch.object(gaia_client, "_query_batch", side_effect=TimeoutError("timeout")),
                patch.object(gaia_client, "_query_individual_batch") as individual,
            ):
                result = gaia_client.match_rows(
                    rows,
                    cache_path,
                    radius_arcsec=1.0,
                    ambiguity_delta_arcsec=0.2,
                    batch_size=50,
                    use_batch=True,
                )

            individual.assert_not_called()
            self.assertEqual(result["match_status"].tolist(), [ERROR, ERROR])

    def test_build_star_object_dataset_without_copying_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            image_path = dataset_root / "images" / "star" / "o1.jpg"
            spectrum_path = dataset_root / "spectra" / "star" / "s1.fits"
            plot_path = dataset_root / "plots" / "star" / "s1_spectrum.png"
            metadata_path = dataset_root / "metadata" / "spectra" / "star" / "s1.json"
            for path in (image_path, spectrum_path, plot_path, metadata_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("placeholder", encoding="utf-8")

            comparison = root / "comparison.csv"
            pd.DataFrame(
                [
                    {
                        "objID": "o1",
                        "specObjID": "s1",
                        "spectralClass": "STAR",
                        "pipeline_subclass": "M3",
                        "ra": 10.0,
                        "dec": -2.0,
                        "match_status": "NO_MATCH",
                        "roman_luminosity_class_raw": float("nan"),
                        "image_path": str(image_path),
                        "spectrum_path": str(spectrum_path),
                        "spectrum_metadata_path": str(metadata_path),
                    }
                ]
            ).to_csv(comparison, index=False)

            output = root / "star_objects"
            index = build_star_object_dataset(comparison, output, dataset_root)
            target_path = output / "STAR_000001" / "Target" / "target.json"
            rgb_path = output / "STAR_000001" / "RGB" / "metadata.json"

            self.assertEqual(index.loc[0, "star_id"], "STAR_000001")
            self.assertTrue(target_path.exists())
            self.assertFalse((output / "STAR_000001" / "RGB" / "image_rgb.jpg").exists())

            target_text = target_path.read_text(encoding="utf-8")
            self.assertNotIn("NaN", target_text)
            target = json.loads(target_text)
            rgb = json.loads(rgb_path.read_text(encoding="utf-8"))
            self.assertEqual(target["spectral_class_major"], "M")
            self.assertEqual(target["roman_luminosity_class_raw"], "")
            self.assertEqual(target["stellar_evolution_class"], "UNKNOWN")
            self.assertEqual(rgb["image_path"], str(image_path))


if __name__ == "__main__":
    unittest.main()
