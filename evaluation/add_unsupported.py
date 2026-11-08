import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app"))

from ingestion import extract_pages_from_pdf

PDF_DIR = Path(r"E:\kg25\RAG-Datasets\financebench-main\pdfs")
DATASET = ROOT / "evaluation" / "dataset_financebench.json"

# (id, filing, question, keywords that must appear together on a page for the question to be answerable)
CANDIDATES = [
    ("UNSUP-4", "AMD_2022_10K", "What was AMD's net revenue in fiscal year 2030?", ["revenue", "2030"]),
    ("UNSUP-5", "AMD_2022_10K", "How many patents did AMD file in Germany in 2022?", ["patents", "germany"]),
    ("UNSUP-6", "BOEING_2022_10K", "What was Boeing's customer satisfaction survey score in 2022?", ["satisfaction", "score"]),
    ("UNSUP-7", "BOEING_2022_10K", "How many engineers did Boeing hire in India in 2022?", ["hired", "india"]),
    ("UNSUP-8", "AMERICANEXPRESS_2022_10K", "What was American Express's market share of Japanese card transactions in 2022?", ["market share", "japan"]),
    ("UNSUP-9", "AMERICANEXPRESS_2022_10K", "How many customer service calls did American Express handle in 2022?", ["customer service", "calls"]),
    ("UNSUP-10", "AMERICANEXPRESS_2022_10K", "What dividend per share will American Express pay in 2030?", ["dividend", "2030"]),
    ("UNSUP-11", "AMCOR_2023_10K", "What is Amcor's forecast net sales for fiscal 2030?", ["net sales", "2030"]),
    ("UNSUP-12", "AMCOR_2023_10K", "How many employees did Amcor have in Brazil in 2023?", ["employees", "brazil"]),
    ("UNSUP-13", "AMCOR_2023_10K", "What was Amcor's customer churn rate in fiscal 2023?", ["churn"]),
]

cases = json.load(open(DATASET, encoding="utf-8"))
existing = {c["id"] for c in cases}
cache = {}
added = 0

for cid, doc, question, keywords in CANDIDATES:
    if cid in existing:
        print(f"{cid}: already in dataset")
        continue
    if doc not in cache:
        print(f"Reading {doc} ...")
        cache[doc] = [p["text"].lower() for p in extract_pages_from_pdf(str(PDF_DIR / f"{doc}.pdf"))]
    hits = [i + 1 for i, t in enumerate(cache[doc]) if all(k in t for k in keywords)]
    if hits:
        print(f"{cid}: SKIPPED, {keywords} appear together on pages {hits[:6]}. Review by hand.")
        continue
    cases.append({"id": cid, "category": "unsupported", "expected_behavior": "refuse",
                  "question": question, "source": f"{doc}.pdf"})
    added += 1
    print(f"{cid}: ADDED ({doc})")

DATASET.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nAdded {added} questions. Unsupported total:", sum(c["category"] == "unsupported" for c in cases))