from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from gaia_sdss import gaia_verified_stars
from gaia_sdss.gaia_verified_stars import (
    _read_cache_csv,
    _write_cache_csv_atomic,
    filter_gaia_verified_candidates,
    gaia_luminosity_class_from_evolstage,
    match_rows_with_gaia_ap,
    parse_gaia_spectral_class,
    prepare_sdss_star_candidates,
    read_sdss_candidate_csv,
)
from gaia_sdss.matching import ERROR, MATCH_AMBIGUOUS, MATCH_UNIQUE, NO_MATCH
from scripts.download_gaia_verified_sdss_stars import filter_outputs_to_verified, trim_verified_to_target


class GaiaVerifiedStarsTests(unittest.TestCase):
    def test_parse_gaia_spectral_class(self):
        self.assertEqual(parse_gaia_spectral_class("G"), "G")
        self.assertEqual(parse_gaia_spectral_class("CSTAR"), "CSTAR")
        self.assertEqual(parse_gaia_spectral_class(""), "")

    def test_gaia_luminosity_class_from_evolstage(self):
        self.assertEqual(gaia_luminosity_class_from_evolstage(100), "MAIN_SEQUENCE")
        self.assertEqual(gaia_luminosity_class_from_evolstage(450), "SUBGIANT")
        self.assertEqual(gaia_luminosity_class_from_evolstage(490), "RED_GIANT_BRANCH")
        self.assertEqual(gaia_luminosity_class_from_evolstage(None), "")

    def test_prepare_candidates_rejects_existing_and_duplicates(self):
        source = pd.DataFrame(
            [
                {"objID": "o1", "specObjID": "s1", "spectralClass": "STAR", "ra": 10.0, "dec": 1.0},
                {"objID": "o2", "specObjID": "s1", "spectralClass": "STAR", "ra": 10.1, "dec": 1.1},
                {"objID": "o1", "specObjID": "s3", "spectralClass": "STAR", "ra": 10.2, "dec": 1.2},
                {"objID": "o4", "specObjID": "s4", "spectralClass": "GALAXY", "ra": 10.3, "dec": 1.3},
                {"objID": "o5", "specObjID": "s5", "spectralClass": "STAR", "ra": None, "dec": 1.4},
                {"objID": "old", "specObjID": "old_spec", "spectralClass": "STAR", "ra": 10.5, "dec": 1.5},
            ]
        )
        existing = {"objID": {"old"}, "specObjID": {"old_spec"}, "source_id": set()}

        selected, rejected = prepare_sdss_star_candidates(source, existing)

        self.assertEqual(selected["specObjID"].tolist(), ["s1"])
        self.assertEqual(
            set(rejected["rejection_reason"]),
            {
                "duplicate_specObjID_in_input",
                "duplicate_objID_in_input",
                "not_sdss_star",
                "missing_required_sdss_identifier_or_position",
                "already_exists_in_local_dataset",
            },
        )

    def test_filter_gaia_verified_candidates_requires_gaia_ap_and_unique_source(self):
        candidates = pd.DataFrame(
            [
                {"objID": "o1", "specObjID": "s1", "spectralClass": "STAR", "ra": 10.0, "dec": 1.0, "has_radec": True, "_source_row": 0},
                {"objID": "o2", "specObjID": "s2", "spectralClass": "STAR", "ra": 11.0, "dec": 2.0, "has_radec": True, "_source_row": 1},
                {"objID": "o3", "specObjID": "s3", "spectralClass": "STAR", "ra": 12.0, "dec": 3.0, "has_radec": True, "_source_row": 2},
                {"objID": "o4", "specObjID": "s4", "spectralClass": "STAR", "ra": 13.0, "dec": 4.0, "has_radec": True, "_source_row": 3},
                {"objID": "o5", "specObjID": "s5", "spectralClass": "STAR", "ra": 14.0, "dec": 5.0, "has_radec": True, "_source_row": 4},
            ]
        )
        matches = pd.DataFrame(
            [
                {
                    "objID": "o1",
                    "specObjID": "s1",
                    "match_status": MATCH_UNIQUE,
                    "best_distance_arcsec": 0.1,
                    "source_id": "gaia1",
                    "spectraltype_esphs": "G",
                    "evolstage_flame": 100,
                    "lum_flame": 1.0,
                },
                {
                    "objID": "o2",
                    "specObjID": "s2",
                    "match_status": MATCH_UNIQUE,
                    "best_distance_arcsec": 0.1,
                    "source_id": "gaia2",
                    "spectraltype_esphs": "",
                    "evolstage_flame": 100,
                    "lum_flame": 1.0,
                },
                {
                    "objID": "o3",
                    "specObjID": "s3",
                    "match_status": NO_MATCH,
                    "best_distance_arcsec": None,
                    "source_id": "",
                    "spectraltype_esphs": "K",
                    "evolstage_flame": 100,
                    "lum_flame": 1.0,
                },
                {
                    "objID": "o4",
                    "specObjID": "s4",
                    "match_status": MATCH_UNIQUE,
                    "best_distance_arcsec": 0.2,
                    "source_id": "gaia1",
                    "spectraltype_esphs": "G",
                    "evolstage_flame": 100,
                    "lum_flame": 1.0,
                },
                {
                    "objID": "o5",
                    "specObjID": "s5",
                    "match_status": MATCH_AMBIGUOUS,
                    "best_distance_arcsec": 0.1,
                    "source_id": "gaia5",
                    "spectraltype_esphs": "M",
                    "evolstage_flame": 490,
                    "lum_flame": 20.0,
                },
            ]
        )

        verified, rejected = filter_gaia_verified_candidates(candidates, matches)

        self.assertEqual(verified["specObjID"].tolist(), ["s1"])
        self.assertEqual(
            set(rejected["rejection_reason"]),
            {
                "MISSING_GAIA_SPECTRAL_CLASS",
                "gaia_match_not_unique_enough",
                "duplicate_gaia_source_id_in_verified_batch",
            },
        )

    def test_trim_verified_to_target_marks_overflow_as_deleted_from_target(self):
        verified = pd.DataFrame(
            [
                {"specObjID": "s1"},
                {"specObjID": "s2"},
                {"specObjID": "s3"},
            ]
        )
        rejected = pd.DataFrame([{"specObjID": "bad", "rejection_reason": "NO_MATCH"}])

        kept, rejected = trim_verified_to_target(verified, rejected, 2)

        self.assertEqual(kept["specObjID"].tolist(), ["s1", "s2"])
        self.assertIn("over_target_gaia_confirmed", set(rejected["rejection_reason"]))

    def test_filter_outputs_to_verified_deletes_unconfirmed_from_clean_outputs(self):
        candidates = pd.DataFrame(
            [
                {"specObjID": "s1", "objID": "o1"},
                {"specObjID": "s2", "objID": "o2"},
                {"specObjID": "s3", "objID": "o3"},
            ]
        )
        matches = pd.DataFrame(
            [
                {"specObjID": "s1", "match_status": MATCH_UNIQUE},
                {"specObjID": "s2", "match_status": NO_MATCH},
            ]
        )
        verified = pd.DataFrame([{"specObjID": "s1", "objID": "o1"}])

        clean_candidates, clean_matches = filter_outputs_to_verified(candidates, matches, verified)

        self.assertEqual(clean_candidates["specObjID"].tolist(), ["s1"])
        self.assertEqual(clean_matches["specObjID"].tolist(), ["s1"])

    def test_empty_candidate_csv_reports_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.csv"
            path.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "CSV SDSS esta vazio"):
                read_sdss_candidate_csv(path)

    def test_retry_errors_also_queries_new_uncached_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "gaia_ap_matches.csv"
            pd.DataFrame(
                [
                    {"specObjID": "s1", "objID": "o1", "match_status": MATCH_UNIQUE, "source_id": "g1"},
                    {"specObjID": "s2", "objID": "o2", "match_status": ERROR},
                ]
            ).to_csv(cache_path, index=False)
            rows = pd.DataFrame(
                [
                    {"specObjID": "s1", "objID": "o1", "ra": 1.0, "dec": 1.0, "has_radec": True},
                    {"specObjID": "s2", "objID": "o2", "ra": 2.0, "dec": 2.0, "has_radec": True},
                    {"specObjID": "s3", "objID": "o3", "ra": 3.0, "dec": 3.0, "has_radec": True},
                ]
            )

            def fake_query(batch, radius_arcsec, query_retries, retry_sleep_seconds, network_timeout):
                self.assertEqual(batch["specObjID"].tolist(), ["s2", "s3"])
                return pd.DataFrame(
                    [
                        {"specObjID": "s2", "objID": "o2", "source_id": "g2", "separation_arcsec": 0.1},
                        {"specObjID": "s3", "objID": "o3", "source_id": "g3", "separation_arcsec": 0.1},
                    ]
                ), {}

            with patch.object(
                gaia_verified_stars,
                "_query_individual_candidates_with_errors",
                side_effect=fake_query,
            ) as query:
                result = match_rows_with_gaia_ap(
                    rows,
                    cache_path,
                    radius_arcsec=1.0,
                    ambiguity_delta_arcsec=0.2,
                    batch_size=50,
                    retry_errors=True,
                    use_batch=False,
                )

            query.assert_called_once()
            by_id = result.set_index("specObjID")
            self.assertEqual(by_id.loc["s1", "match_status"], MATCH_UNIQUE)
            self.assertEqual(by_id.loc["s2", "match_status"], MATCH_UNIQUE)
            self.assertEqual(by_id.loc["s3", "match_status"], MATCH_UNIQUE)

    def test_cache_read_falls_back_to_backup_after_corrupt_main_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "gaia_ap_matches.csv"
            original = pd.DataFrame(
                [{"specObjID": "s1", "objID": "o1", "match_status": MATCH_UNIQUE}]
            )
            updated = pd.DataFrame(
                [
                    {"specObjID": "s1", "objID": "o1", "match_status": MATCH_UNIQUE},
                    {"specObjID": "s2", "objID": "o2", "match_status": NO_MATCH},
                ]
            )

            _write_cache_csv_atomic(cache_path, original)
            _write_cache_csv_atomic(cache_path, updated)
            cache_path.write_text('"unterminated', encoding="utf-8")

            recovered = _read_cache_csv(cache_path)

            self.assertEqual(recovered["specObjID"].tolist(), ["s1"])


if __name__ == "__main__":
    unittest.main()
