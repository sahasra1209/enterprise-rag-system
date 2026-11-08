import json
import re
from pathlib import Path

import chromadb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(r"E:\kg25\RAG-Datasets\financebench-main\data\financebench_open_source.jsonl")
OUT = ROOT / "evaluation" / "dataset_financebench.json"

DOCS = [
    "AMD_2022_10K", "AMERICANEXPRESS_2022_10K", "BOEING_2022_10K", "PEPSICO_2022_10K",
    "AMCOR_2023_10K", "3M_2022_10K", "BESTBUY_2023_10K", "CVSHEALTH_2022_10K",
]

# Starter cases. Grow these to ~10 each. Confirm each unsupported question
# really isn't answered in the PDF (Ctrl+F) before keeping it.
EXTRA_CASES = [
    {"id": "UNSUP-1", "category": "unsupported", "expected_behavior": "refuse",
     "question": "Who will be Boeing's CEO in 2035?", "source": "BOEING_2022_10K.pdf"},
    {"id": "UNSUP-2", "category": "unsupported", "expected_behavior": "refuse",
     "question": "What was PepsiCo's customer satisfaction score in 2022?", "source": "PEPSICO_2022_10K.pdf"},
    {"id": "UNSUP-3", "category": "unsupported", "expected_behavior": "refuse",
     "question": "What will AMD's total revenue be in fiscal year 2030?", "source": "AMD_2022_10K.pdf"},
    {"id": "CONTRA-1", "category": "contradictory", "expected_behavior": "flag_contradiction",
     "question": "AMD's balance sheet shows total current liabilities of $9,369 million at December 31, 2022. Is that correct?",
     "reference_answer": "No. Total current liabilities were $6,369 million.", "source": "AMD_2022_10K.pdf"},
    {"id": "CONTRA-2", "category": "contradictory", "expected_behavior": "flag_contradiction",
     "question": "AMD reported total current assets of $12,019 million at December 31, 2022, correct?",
     "reference_answer": "No. Total current assets were $15,019 million.", "source": "AMD_2022_10K.pdf"},
]

col = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection("policy_documents")


def page_numbers_text(source, page):
    res = col.get(where={"$and": [{"source": source}, {"page_number": page}]}, include=["documents"])
    return " ".join(res["documents"])


def nums(text):
    return set(re.findall(r"\d[\d,]{3,}", text))


df = pd.read_json(SRC, lines=True)
df = df[df.doc_name.isin(DOCS)]

# 1. Work out the page offset by testing several values against the real index.
offsets = [-1, 0, 1, 2]
score = {o: 0 for o in offsets}
checked = 0
for _, r in df.iterrows():
    for e in (r["evidence"] or []):
        ev_nums = nums(e["evidence_text"])
        if not ev_nums:
            continue
        checked += 1
        for o in offsets:
            found = nums(page_numbers_text(f"{e['doc_name']}.pdf", int(e["evidence_page_num"]) + o))
            if len(ev_nums & found) / len(ev_nums) >= 0.5:
                score[o] += 1
print(f"Evidence items checked: {checked}")
for o in offsets:
    print(f"  offset {o:+d}: {score[o]} matches")
PAGE_OFFSET = max(score, key=score.get)
print("Using page offset:", PAGE_OFFSET)

# 2. Build the supported cases.
cases = []
for _, r in df.iterrows():
    ev = r["evidence"] or []
    reasoning = str(r["question_reasoning"]).strip().lower()
    cases.append({
        "id": r["financebench_id"],
        "category": "supported",
        "expected_behavior": "answer",
        "tier": "extraction" if reasoning == "information extraction" else "reasoning",
        "question": r["question"],
        "reference_answer": r["answer"],
        "source": f"{r['doc_name']}.pdf",
        "evidence_pages": sorted({int(e["evidence_page_num"]) + PAGE_OFFSET for e in ev}),
        "evidence_text": [e["evidence_text"] for e in ev],
    })

cases.extend(EXTRA_CASES)

OUT.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(cases)} cases to {OUT}")
for cat in ("supported", "unsupported", "contradictory"):
    print(f"  {cat}: {sum(c['category'] == cat for c in cases)}")