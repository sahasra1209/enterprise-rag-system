import json
import re
import sys
from pathlib import Path

import ollama
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app"))

from answer import FALLBACK, get_evidence_context

SRC = Path(r"E:\kg25\RAG-Datasets\financebench-main\data\financebench_open_source.jsonl")
DATASET = ROOT / "evaluation" / "dataset_financebench.json"
OUT = ROOT / "evaluation" / "results" / "llm_probe.json"
PAGE_OFFSET = 1
MAX_PASSAGES = 2
MAX_CHARS = 1800
N_CASES = 15  # extraction questions first, then reasoning

# Variant A: copied from backend/app/answer.py
PROMPT_A = """
You are a policy-document question-answering assistant.

Answer the user's question using ONLY the supplied evidence.

Instructions:
- If the evidence explicitly answers the question, answer directly.
- Do not refuse a question that the evidence clearly answers.
- Do not invent facts.
- Do not add information absent from the evidence.
- Cite every factual sentence using an evidence ID such as [E1].
- Put citations before the final punctuation.
- Use only citation IDs present in the supplied evidence.
- If the evidence does not answer the question, return exactly:

{fallback}

Evidence:
{context}

Question:
{question}

Answer:
""".strip()

# Variant B: same rules, reworded for financial filings
PROMPT_B = """
You answer questions about company annual reports (10-K filings).

Use ONLY the evidence below. Write a short answer in your own words, and cite the
evidence ID such as [E1] after each sentence. Numbers and ratios may be computed
from figures in the evidence. If the evidence truly contains nothing relevant,
reply exactly: {fallback}

Evidence:
{context}

Question:
{question}

Answer:
""".strip()


def nums(t):
    return {n.replace(",", "").rstrip(".") for n in re.findall(r"\d[\d,]*\.?\d*", str(t))}


raw = pd.read_json(SRC, lines=True).set_index("financebench_id")
cases = [c for c in json.load(open(DATASET, encoding="utf-8")) if c["category"] == "supported"]
cases.sort(key=lambda c: c["tier"] != "extraction")
cases = cases[:N_CASES]

rows = []
for c in cases:
    ev = list(raw.loc[c["id"], "evidence"])[:MAX_PASSAGES]
    evidence = {
        f"E{i}": {"text": e["evidence_text"][:MAX_CHARS], "source": f"{e['doc_name']}.pdf", "page": int(e["evidence_page_num"]) + PAGE_OFFSET}
        for i, e in enumerate(ev, 1)
    }
    context = get_evidence_context(evidence)
    row = {"id": c["id"], "tier": c["tier"], "question": c["question"], "reference": c["reference_answer"]}
    for name, tmpl in (("A", PROMPT_A), ("B", PROMPT_B)):
        prompt = tmpl.format(fallback=FALLBACK, context=context, question=c["question"])
        out = ollama.chat(model="qwen2.5:3b", messages=[{"role": "user", "content": prompt}],
                          options={"temperature": 0})["message"]["content"].strip()
        row[name] = out
        row[name + "_refused"] = FALLBACK.lower() in out.lower()
        ref_n = nums(c["reference_answer"])
        row[name + "_num_ok"] = (len(ref_n & nums(out)) / len(ref_n) >= 0.5) if ref_n and not row[name + "_refused"] else None
    rows.append(row)
    print(f"{c['id']} | {c['tier']:<10} | A refused={row['A_refused']!s:<5} | B refused={row['B_refused']!s:<5}")

OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\nTotal cases: {len(rows)}")
for name in ("A", "B"):
    refused = sum(r[name + "_refused"] for r in rows)
    graded = [r for r in rows if r[name + "_num_ok"] is not None]
    print(f"Prompt {name}: refused {refused}/{len(rows)}; numbers match {sum(r[name + '_num_ok'] for r in graded)}/{len(graded)} of answered questions with numbers")

print("\n--- First 4 raw answers (prompt A vs B) ---")
for r in rows[:4]:
    print("\nQ:", r["question"][:120])
    print("REF:", r["reference"][:120])
    print("A  :", r["A"][:250])
    print("B  :", r["B"][:250])
print("\nSaved:", OUT)