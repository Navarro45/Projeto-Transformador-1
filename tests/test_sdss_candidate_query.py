from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.query_sdss_dr20_star_candidates import (
    OUTPUT_COLUMNS,
    build_star_candidate_query,
    max_specobjid,
    parse_skyserver_csv,
    read_existing_output,
)


class SdssCandidateQueryTests(unittest.TestCase):
    def test_build_star_candidate_query_uses_dr20_candidate_filters(self):
        query = build_star_candidate_query(
            batch_size=123,
            min_specobjid=456,
            include_zwarning=False,
            include_non_clean_photometry=False,
            allow_any_photo_type=False,
        )

        self.assertIn("SELECT TOP 123", query)
        self.assertIn("s.class = 'STAR'", query)
        self.assertIn("s.sciencePrimary = 1", query)
        self.assertIn("s.specObjID > 456", query)
        self.assertIn("s.zWarning = 0", query)
        self.assertIn("p.clean = 1", query)
        self.assertIn("p.type = 6", query)

    def test_build_star_candidate_query_can_relax_quality_filters(self):
        query = build_star_candidate_query(
            batch_size=10,
            min_specobjid=0,
            include_zwarning=True,
            include_non_clean_photometry=True,
            allow_any_photo_type=True,
        )

        self.assertNotIn("s.zWarning = 0", query)
        self.assertNotIn("p.clean = 1", query)
        self.assertNotIn("p.type = 6", query)

    def test_parse_skyserver_csv_normalizes_expected_columns(self):
        text = "\n".join(
            [
                "objid,ra,dec,type,specobjid,spectralclass,redshift,redshifterror,zwarning,plate,mjd,fiberid",
                "1,10.5,-1.2,6,99,STAR,0.0,0.0,0,100,50000,20",
            ]
        )

        frame = parse_skyserver_csv(text)

        self.assertEqual(frame.columns.tolist(), OUTPUT_COLUMNS)
        self.assertEqual(str(frame.loc[0, "spectralClass"]), "STAR")
        self.assertEqual(int(frame.loc[0, "specObjID"]), 99)

    def test_parse_skyserver_csv_rejects_error_response(self):
        with self.assertRaisesRegex(RuntimeError, "SQL syntax"):
            parse_skyserver_csv("Error: SQL syntax failed")

    def test_read_existing_output_handles_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.csv"
            path.write_text("", encoding="utf-8")

            frame = read_existing_output(path)

        self.assertTrue(frame.empty)
        self.assertEqual(frame.columns.tolist(), OUTPUT_COLUMNS)

    def test_max_specobjid_uses_numeric_values(self):
        frame = pd.DataFrame({"specObjID": ["10", "25", "bad"]})
        self.assertEqual(max_specobjid(frame), 25)


if __name__ == "__main__":
    unittest.main()
