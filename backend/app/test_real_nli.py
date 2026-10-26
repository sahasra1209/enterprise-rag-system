
from hallucination_detector import HallucinationDetector

detector = HallucinationDetector()

tests = [
    {
        "name": "Supported",
        "claim": "Employees may work remotely two days per week.",
        "evidence": "Employees may work remotely two days per week.",
        "expected": "supported",
    },
    {
        "name": "Contradicted",
        "claim": "Employees may work remotely five days per week.",
        "evidence": "Employees may work remotely two days per week.",
        "expected": "contradicted",
    },
    {
        "name": "Not established",
        "claim": "Employees receive 12 sick-leave days per year.",
        "evidence": "Sick leave is available for eligible employees.",
        "expected": "not_established",
    },
]

for test in tests:
    result = detector.verify_claim(
        test["claim"],
        test["evidence"],
    )

    status = result["status"]
    passed = status == test["expected"]

    print(f"\nTest: {test['name']}")
    print(f"Expected: {test['expected']}")
    print(f"Actual:   {status}")
    print(f"Result:   {'PASS' if passed else 'FAIL'}")
    print(f"Details:  {result}")
