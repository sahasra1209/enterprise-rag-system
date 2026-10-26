
import re
from sentence_transformers import CrossEncoder


class HallucinationDetector:
    """Verify answer claims against evidence using NLI."""

    def __init__(
        self,
        model_name="cross-encoder/nli-deberta-v3-small",
        entailment_threshold=0.70,
        contradiction_threshold=0.70,
    ):
        if not 0.0 <= entailment_threshold <= 1.0:
            raise ValueError(
                "Entailment threshold must be between 0 and 1."
            )

        if not 0.0 <= contradiction_threshold <= 1.0:
            raise ValueError(
                "Contradiction threshold must be between 0 and 1."
            )

        self.model = CrossEncoder(model_name)
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold

        config_labels = self.model.model.config.id2label

        self.labels = {
            int(index): str(label).lower()
            for index, label in config_labels.items()
        }

        # Handle generic model labels.
        if any(
            label.startswith("label_")
            for label in self.labels.values()
        ):
            self.labels = {
                0: "contradiction",
                1: "entailment",
                2: "neutral",
            }

        required_labels = {
            "contradiction",
            "entailment",
            "neutral",
        }

        if not required_labels.issubset(set(self.labels.values())):
            raise ValueError(
                f"Unexpected NLI model labels: {self.labels}"
            )

    @staticmethod
    def split_evidence(evidence: str) -> list[str]:
        """Split evidence into individual sentences."""
        sentences = re.split(
            r"(?<=[.!?])\s+|\n+",
            evidence.strip(),
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def verify_claim(self, claim: str, evidence: str) -> dict:
        """Return supported, contradicted, or not_established."""

        if not claim or not claim.strip():
            raise ValueError("Claim cannot be empty.")

        if not evidence or not evidence.strip():
            return {
                "status": "not_established",
                "scores": {},
                "reason": "No supporting evidence was provided.",
            }

        claim = claim.strip()
        sentences = self.split_evidence(evidence)

        if not sentences:
            return {
                "status": "not_established",
                "scores": {},
                "reason": "No usable evidence sentences were found.",
            }

        # Compare each evidence sentence with the claim.
        # Premise = evidence; hypothesis = claim.
        pairs = [(sentence, claim) for sentence in sentences]

        probabilities = self.model.predict(
            pairs,
            apply_softmax=True,
            convert_to_numpy=True,
        )

        sentence_results = []
        print("\n--- NLI DEBUG ---")
        print("CLAIM:", claim)

        for item in sentence_results:
            print("\nEVIDENCE:", item["evidence_sentence"])
            print("SCORES:", item["scores"])

        for sentence, scores_array in zip(sentences, probabilities):
            scores = {
                self.labels[index]: float(scores_array[index])
                for index in range(len(scores_array))
            }

            sentence_results.append({
                "evidence_sentence": sentence,
                "scores": scores,
            })

        # Step 1: Find the strongest supporting sentence.
        best_support = max(
            sentence_results,
            key=lambda item: item["scores"]["entailment"],
        )

        support_score = best_support["scores"]["entailment"]

        # Prioritize strong direct support over unrelated sentences
        # that may be incorrectly classified as contradictions.
        if support_score >= self.entailment_threshold:
            return {
                "status": "supported",
                "scores": best_support["scores"],
                "matched_evidence": best_support["evidence_sentence"],
                "reason": "An evidence sentence supports the claim.",
            }

        # Step 2: Check for contradiction if strong support is absent.
        best_contradiction = max(
            sentence_results,
            key=lambda item: item["scores"]["contradiction"],
        )

        contradiction_score = (
            best_contradiction["scores"]["contradiction"]
        )

        if contradiction_score >= self.contradiction_threshold:
            return {
                "status": "contradicted",
                "scores": best_contradiction["scores"],
                "matched_evidence": (
                    best_contradiction["evidence_sentence"]
                ),
                "reason": "An evidence sentence may conflict with the claim.",
            }

        # Step 3: Neither support nor contradiction is sufficiently strong.
        return {
            "status": "not_established",
            "scores": best_support["scores"],
            "matched_evidence": best_support["evidence_sentence"],
            "reason": (
                "The evidence does not establish the claim confidently."
            ),
        }
