import csv
from collections import Counter
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "evaluation" / "results" / "error_analysis_promptD.csv"
rows = list(csv.DictReader(open(path, encoding="utf-8-sig", errors="replace", newline="")))
labeled = [r for r in rows if r["your_label"].strip() and r["stage"].startswith("answered")]
print("Answered rows:", sum(r["stage"].startswith("answered") for r in rows), "| labeled:", len(labeled))
counts = Counter(r["your_label"].strip().lower() for r in labeled)
for k, v in counts.most_common():
    print(f"  {k}: {v}")
for r in labeled:
    print(f"{r['id']}: {r['your_label'].strip()}  | {r['your_notes']}")