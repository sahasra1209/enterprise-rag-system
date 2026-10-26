
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from hallucination_detector import HallucinationDetector


class TestHallucinationDetector(unittest.TestCase):

    def setUp(self):
        patcher = patch("hallucination_detector.CrossEncoder")
        self.mock_cross_encoder = patcher.start()
        self.addCleanup(patcher.stop)

        fake_model = MagicMock()
        fake_model.model.config.id2label = {
            0: "contradiction",
            1: "entailment",
            2: "neutral",
        }

        # Default prediction: the claim is supported.
        fake_model.predict.return_value = np.array([
            [0.01, 0.98, 0.01]
        ])

        self.mock_cross_encoder.return_value = fake_model
        self.fake_model = fake_model
        self.detector = HallucinationDetector()

    def test_supported_claim(self):
        """Evidence directly supports the claim."""
        result = self.detector.verify_claim(
            "Employees receive 20 days of annual leave per year.",
            "Employees receive 20 days of annual leave per year."
        )

        self.assertEqual(result["status"], "supported")

    def test_remote_work_claim_supported(self):
        """The policy explicitly allows two remote-work days."""
        result = self.detector.verify_claim(
            "Employees may work remotely two days per week.",
            "Employees may work remotely two days per week."
        )

        self.assertEqual(result["status"], "supported")

    def test_remote_work_claim_contradicted(self):
        """The claim says five days, but the policy says two."""
        self.fake_model.predict.return_value = np.array([
            [0.98, 0.01, 0.01]
        ])

        result = self.detector.verify_claim(
            "Employees may work remotely five days per week.",
            "Employees may work remotely two days per week."
        )

        self.assertEqual(result["status"], "contradicted")

    def test_claim_not_established(self):
        """The evidence does not specify a sick-leave allowance."""
        self.fake_model.predict.return_value = np.array([
            [0.01, 0.01, 0.98]
        ])

        result = self.detector.verify_claim(
            "Employees receive 12 sick-leave days per year.",
            "Sick leave is available for eligible employees."
        )

        self.assertEqual(result["status"], "not_established")

    def test_empty_evidence(self):
        """Empty evidence cannot establish a claim."""
        result = self.detector.verify_claim(
            "Employees receive 20 days of leave.",
            ""
        )

        self.assertEqual(result["status"], "not_established")

    def test_empty_claim_raises_error(self):
        """An empty claim should raise ValueError."""
        with self.assertRaises(ValueError):
            self.detector.verify_claim("", "Some evidence.")

    def test_empty_evidence_does_not_call_model(self):
        """The model should not run when evidence is empty."""
        self.fake_model.predict.reset_mock()

        self.detector.verify_claim("A claim.", "  ")

        self.fake_model.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
