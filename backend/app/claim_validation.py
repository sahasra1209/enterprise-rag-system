
import re


def normalize(text: str) -> set[str]:
    """Extract normalized words."""
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


def extract_numbers(text: str) -> set[str]:
    """Extract numeric values from text."""
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text.lower()))


def contains_negation(text: str) -> bool:
    """Detect common negation words."""
    negations = {
        "not", "no", "never", "cannot", "can't",
        "isn't", "aren't", "doesn't", "don't",
        "didn't", "won't", "without"
    }

    words = set(re.findall(r"\b[\w']+\b", text.lower()))
    return bool(words & negations)


def validate_claim(claim: str, evidence: str) -> bool:
    """Conservative lexical baseline, not semantic proof."""

    if not claim or not claim.strip():
        return False

    if not evidence or not evidence.strip():
        return False

    claim_words = normalize(claim)
    evidence_words = normalize(evidence)

    if not claim_words or not evidence_words:
        return False

    # Reject claims whose numeric values conflict with the evidence.
    claim_numbers = extract_numbers(claim)
    evidence_numbers = extract_numbers(evidence)

    if claim_numbers and claim_numbers != evidence_numbers:
        return False

    # Avoid treating opposite polarity as supported.
    if contains_negation(claim) != contains_negation(evidence):
        return False

    overlap = claim_words & evidence_words
    overlap_ratio = len(overlap) / len(claim_words)

    return overlap_ratio >= 0.5


if __name__ == "__main__":
    examples = [
        (
            "Managers must approve leave requests.",
            "Managers must approve leave requests.",
            True,
        ),
        (
            "Employees receive 30 days of annual leave.",
            "Employees receive 20 days of annual leave.",
            False,
        ),
        (
            "Maternity leave is 12 weeks.",
            "Employees can request leave through the HR portal.",
            False,
        ),
        (
            "Managers must approve leave requests.",
            "Managers do not approve leave requests.",
            False,
        ),
    ]

    passed = 0

    for claim, evidence, expected in examples:
        actual = validate_claim(claim, evidence)
        status = "PASS" if actual == expected else "FAIL"
        print(f"{status}: {claim} -> {actual}")

        if actual == expected:
            passed += 1

    print(f"\nValidation tests: {passed}/{len(examples)} passed")
