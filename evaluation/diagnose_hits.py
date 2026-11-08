import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cases = {c["id"]: c for c in json.load(open(ROOT / "evaluation" / "dataset_financebench.json", encoding="utf-8"))}
runs = json.load(open(ROOT / "evaluation" / "results" / "run_baseline.json", encoding="utf-8"))


def nums(t):
    return {n.replace(",", "").rstrip(".") for n in re.findall(r"\d[\d,]*\.?\d*", str(t))}


for run in runs:
    c = cases[run["id"]]
    chunks = run["result"].get("retrieved_chunks", [])
    hit = [ch for ch in chunks if ch.get("source") == c["source"] and ch.get("page") in c.get("evidence_pages", [])]
    if not hit:
        continue
    ref = nums(c["reference_answer"])
    found = ref & nums(" ".join(ch["text"] for ch in chunks))
    print("=" * 70)
    print(c["id"], "|", c["question"])
    print("REF:", c["reference_answer"][:200])
    print("reference numbers found in the retrieved chunks:", sorted(found), "of", sorted(ref))
    for ch in hit:
        print("  HIT chunk (page", ch["page"], "):", ch["text"][:300])