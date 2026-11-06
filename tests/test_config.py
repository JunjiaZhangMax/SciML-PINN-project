"""Loss-weight compatibility and experiment provenance."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from sciml.config import Config, suite


WEIGHTS = ("w_observation", "w_initial", "w_boundary", "w_residual", "w_gauge")


class ConfigWeightTests(unittest.TestCase):
    def test_legacy_config_uses_equal_weights(self):
        legacy = Config().to_dict()
        for key in WEIGHTS:
            legacy.pop(key)
        config = Config.from_dict(legacy)
        self.assertTrue(all(getattr(config, key) == 1.0 for key in WEIGHTS))

    def test_nonfinite_or_negative_weights_rejected(self):
        for key in WEIGHTS:
            for value in (-0.1, float("nan"), float("inf"), float("-inf")):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    Config.from_dict({key: value})

    def test_weights_saved_loaded_and_fingerprinted(self):
        config = Config(w_observation=2.0, w_initial=3.0, w_boundary=4.0,
                        w_residual=0.5, w_gauge=0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config.save(path)
            restored = Config.load(path)
        self.assertEqual(config, restored)
        self.assertEqual(config.fingerprint, restored.fingerprint)
        for key in WEIGHTS:
            with self.subTest(key=key):
                self.assertNotEqual(config.fingerprint,
                                    replace(config, **{key: getattr(config, key) + 1}).fingerprint)

    def test_suite_preserves_custom_weights(self):
        base = Config(w_observation=2, w_initial=3, w_boundary=4, w_residual=5, w_gauge=0)
        experiments = suite(base)
        self.assertEqual(len(experiments), 25)
        for config in experiments:
            self.assertEqual({key: getattr(config, key) for key in WEIGHTS},
                             {key: getattr(base, key) for key in WEIGHTS})
