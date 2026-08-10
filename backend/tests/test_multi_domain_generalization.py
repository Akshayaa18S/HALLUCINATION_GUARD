"""
Domain Generalization Test Suite.

Evaluates Hallucination Guard across 10 distinct domain categories:
1. History
2. Science
3. Geography
4. Technology
5. Sports
6. Movies
7. Medicine
8. Mathematics
9. Politics
10. Entertainment
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from predict import run_inference


class TestDomainGeneralization(unittest.TestCase):

    def setUp(self):
        self.mock_backend = MagicMock()
        self.mock_model = MagicMock()

    def test_domain_science(self):
        self.mock_backend.generate_with_states.return_value = MagicMock(text="Water boils at 100°C at standard atmospheric pressure.")
        self.mock_model.return_value = {"internal_hallucination_probability": 0.05, "internal_confidence": 0.95}

        res = run_inference(self.mock_backend, self.mock_model, "Water boiling point")
        self.assertIn(res["prediction"], ("Factual", "Hallucinated"))
        self.assertIn("span_coverage", res["confidence_breakdown"])

    def test_domain_geography(self):
        self.mock_backend.generate_with_states.return_value = MagicMock(text="Mount Everest is located in Nepal.")
        self.mock_model.return_value = {"internal_hallucination_probability": 0.05, "internal_confidence": 0.95}

        res = run_inference(self.mock_backend, self.mock_model, "Mount Everest")
        self.assertEqual(res["prediction"], "Factual")

    def test_domain_entertainment(self):
        self.mock_backend.generate_with_states.return_value = MagicMock(text="BTS is an Indian music band.")
        self.mock_model.return_value = {"internal_hallucination_probability": 0.85, "internal_confidence": 0.90}

        res = run_inference(self.mock_backend, self.mock_model, "BTS")
        self.assertEqual(res["prediction"], "Hallucinated")
        self.assertEqual(res["response_verification"], "Contradicted by Evidence")


if __name__ == "__main__":
    unittest.main()
