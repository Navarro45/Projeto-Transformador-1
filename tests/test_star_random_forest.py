from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from models.model_factory import ModelFactory
from models.random_forest_modelo import (
    FlatRandomForestModel,
    HierarchicalRandomForestModel,
    RandomForestModelConfig,
)
from shared.pipelines.star_supervised_random_forest import (
    build_hierarchy,
    evaluate_probability_outputs,
    extract_spectrum_features,
    hierarchical_distance,
    load_or_build_spectral_features,
    make_split_manifest,
    normalize_spectral_label,
    rare_class_report,
)
from spectral.preprocessing import SpectralPreprocessor, SpectralPreprocessorConfig


def _small_config(seed: int = 7) -> RandomForestModelConfig:
    return RandomForestModelConfig(
        n_estimators=8,
        criterion="gini",
        random_state=seed,
        n_jobs=1,
        class_weight=None,
    )


def _synthetic_supervised_data():
    X = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.2],
            [1.0, 1.0],
            [1.1, 1.0],
            [2.0, 2.0],
            [2.1, 2.0],
            [3.0, 3.0],
            [3.0, 3.2],
        ],
        dtype=np.float32,
    )
    y_parent = np.asarray(["A", "A", "F", "F", "M", "M", "WD", "WD"], dtype=str)
    y_leaf = np.asarray(["A0", "A1", "F0", "F1", "M0", "M1", "WD", "WD"], dtype=str)
    return X, y_parent, y_leaf


def _write_fits(path: Path, phase: float = 0.0) -> None:
    wavelength = np.linspace(3800.0, 9200.0, 96, dtype=np.float32)
    flux = (np.sin(np.linspace(0.0, 8.0, 96) + phase) + np.linspace(1.0, 2.0, 96)).astype(np.float32)
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="flux", array=flux, format="E"),
            fits.Column(name="loglam", array=np.log10(wavelength).astype(np.float32), format="E"),
        ],
        name="COADD",
    )
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


class StarRandomForestLabelTests(unittest.TestCase):
    def test_normalize_spectral_labels(self):
        cases = {
            "M3": ("M", "M3"),
            "F5": ("F", "F5"),
            "A0": ("A", "A0"),
            "A1V": ("A", "A1"),
            "A1III": ("A", "A1"),
            "F3/F5V (30743)": ("F", "F3"),
            "K5Ve (118100)": ("K", "K5"),
            "M4.5:III (123657)": ("M", "M4.5"),
            "WD": ("WD", "WD"),
            "WDhotter": ("WD", "WD"),
            "Carbon_lines": ("CARBON", "CARBON"),
            "CV": ("CV", "CV"),
            "L9": ("L", "L9"),
            "T2": ("T", "T2"),
            "sd:F0 (G_84-29)": ("F", "F0"),
        }
        for raw, expected in cases.items():
            labels = normalize_spectral_label(raw)
            self.assertEqual(
                (labels["spectral_parent_label"], labels["spectral_leaf_label"]),
                expected,
            )

    def test_hierarchy_distance_uses_common_ancestor(self):
        records = pd.DataFrame(
            [
                {"spectral_parent_label": "G", "spectral_leaf_label": "G2"},
                {"spectral_parent_label": "G", "spectral_leaf_label": "G5"},
                {"spectral_parent_label": "M", "spectral_leaf_label": "M3"},
            ]
        )
        hierarchy = build_hierarchy(records)
        self.assertEqual(hierarchical_distance("G2", "G2", hierarchy), 0)
        self.assertEqual(hierarchical_distance("G2", "G5", hierarchy), 2)
        self.assertEqual(hierarchical_distance("G2", "M3", hierarchy), 4)


class StarRandomForestModelTests(unittest.TestCase):
    def test_model_factory_recognizes_sklearn_models(self):
        self.assertTrue(ModelFactory.is_sklearn_model("flat_random_forest"))
        self.assertTrue(ModelFactory.is_sklearn_model("hierarchical_random_forest"))
        self.assertIsInstance(ModelFactory.create_sklearn("flat_random_forest", _small_config()), FlatRandomForestModel)
        self.assertIsInstance(
            ModelFactory.create_sklearn("hierarchical_random_forest", _small_config()),
            HierarchicalRandomForestModel,
        )
        self.assertIn("resnet", ModelFactory.MODELS)

    def test_flat_random_forest_probabilities_are_aligned(self):
        X, _, y_leaf = _synthetic_supervised_data()
        model = FlatRandomForestModel(_small_config()).fit(X, y_leaf)
        labels = sorted(set(y_leaf))
        probabilities = model.predict_leaf_proba(X, labels)
        self.assertEqual(probabilities.shape, (len(X), len(labels)))
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(len(X)))

    def test_hierarchical_random_forest_probabilities_and_paths(self):
        X, y_parent, y_leaf = _synthetic_supervised_data()
        model = HierarchicalRandomForestModel(_small_config()).fit(X, y_leaf, y_parent)
        labels = sorted(set(y_leaf))
        probabilities = model.predict_leaf_proba(X, labels)
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(len(X)))

        paths = model.predict_paths(X)
        for item in paths:
            self.assertEqual(item["path"][0], "STAR")
            self.assertEqual(model.leaf_to_parent[item["leaf"]], item["parent"])

    def test_hierarchical_random_forest_deterministic_child(self):
        X = np.asarray([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
        y_parent = np.asarray(["WD", "WD", "F", "F"], dtype=str)
        y_leaf = np.asarray(["WD", "WD", "F0", "F1"], dtype=str)
        model = HierarchicalRandomForestModel(_small_config()).fit(X, y_leaf, y_parent)
        self.assertEqual(model.deterministic_children["WD"], "WD")
        probabilities = model.predict_leaf_proba(X, sorted(set(y_leaf)))
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(len(X)))

    def test_save_and_load_model(self):
        X, y_parent, y_leaf = _synthetic_supervised_data()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            model = HierarchicalRandomForestModel(_small_config()).fit(X, y_leaf, y_parent)
            model.save(path)
            loaded = HierarchicalRandomForestModel.load(path)
            np.testing.assert_array_equal(model.predict_leaf(X), loaded.predict_leaf(X))

    def test_hierarchical_metrics(self):
        records = pd.DataFrame(
            [
                {"spectral_parent_label": "G", "spectral_leaf_label": "G2"},
                {"spectral_parent_label": "G", "spectral_leaf_label": "G5"},
                {"spectral_parent_label": "M", "spectral_leaf_label": "M3"},
            ]
        )
        hierarchy = build_hierarchy(records)
        parent_labels = hierarchy["parent_labels"]
        leaf_labels = hierarchy["leaf_labels"]
        parent_probabilities = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8]])
        leaf_probabilities = np.asarray([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
        metrics = evaluate_probability_outputs(
            records["spectral_parent_label"].to_numpy(),
            records["spectral_leaf_label"].to_numpy(),
            parent_probabilities,
            leaf_probabilities,
            parent_labels,
            leaf_labels,
            hierarchy,
            alpha=0.35,
        )
        self.assertEqual(metrics["hierarchical"]["cross_parent_error"], 0)
        self.assertGreater(metrics["hierarchical"]["hierarchical_loss"], 0.0)


class StarRandomForestDataTests(unittest.TestCase):
    def test_split_manifest_keeps_singleton_classes_in_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.csv"
            records = pd.DataFrame(
                [
                    {"star_id": "s1", "objID": "o1", "specObjID": "sp1", "spectral_parent_label": "A", "spectral_leaf_label": "A0"},
                    {"star_id": "s2", "objID": "o2", "specObjID": "sp2", "spectral_parent_label": "F", "spectral_leaf_label": "F0"},
                    {"star_id": "s3", "objID": "o3", "specObjID": "sp3", "spectral_parent_label": "F", "spectral_leaf_label": "F0"},
                    {"star_id": "s4", "objID": "o4", "specObjID": "sp4", "spectral_parent_label": "F", "spectral_leaf_label": "F0"},
                ]
            )
            split = make_split_manifest(records, path, seed=42, refresh_split=True)
            singleton = split[split["spectral_leaf_label"].eq("A0")]
            self.assertEqual(singleton.iloc[0]["split"], "train")
            self.assertTrue(path.exists())

    def test_rare_class_report_preserves_all_classes(self):
        records = pd.DataFrame(
            [
                {"spectral_parent_label": "A", "spectral_leaf_label": "A0"},
                {"spectral_parent_label": "F", "spectral_leaf_label": "F0"},
                {"spectral_parent_label": "F", "spectral_leaf_label": "F0"},
            ]
        )
        report = rare_class_report(records, threshold=1)
        self.assertEqual(set(report["spectral_leaf_label"]), {"A0", "F0"})
        self.assertTrue(bool(report.loc[report["spectral_leaf_label"].eq("A0"), "is_rare"].iloc[0]))

    def test_extract_spectrum_features_from_local_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            fits_path = Path(tmp) / "spectrum.fits"
            _write_fits(fits_path)
            config = SpectralPreprocessorConfig(spectral_length=32)
            vector, quality = extract_spectrum_features(fits_path, SpectralPreprocessor(config))
            self.assertEqual(vector.shape, (32,))
            self.assertGreater(quality["coverage_fraction"], 0.9)

    def test_load_or_build_spectral_features_uses_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fits_path = root / "spectrum.fits"
            _write_fits(fits_path)
            records = pd.DataFrame(
                [
                    {
                        "star_id": "STAR_000001",
                        "objID": "o1",
                        "specObjID": "sp1",
                        "spectral_subclass_raw": "G2",
                        "spectral_parent_label": "G",
                        "spectral_leaf_label": "G2",
                        "spectrum_path": str(fits_path),
                    }
                ]
            )
            config = SpectralPreprocessorConfig(spectral_length=16)
            X1, metadata1, failures1, cache1 = load_or_build_spectral_features(records, config, root / "cache")
            X2, metadata2, failures2, cache2 = load_or_build_spectral_features(records, config, root / "cache")

            self.assertFalse(cache1["cache_hit"])
            self.assertTrue(cache2["cache_hit"])
            np.testing.assert_allclose(X1, X2)
            self.assertEqual(metadata1["star_id"].tolist(), metadata2["star_id"].tolist())
            self.assertTrue(failures1.empty)
            self.assertTrue(failures2.empty)


if __name__ == "__main__":
    unittest.main()
