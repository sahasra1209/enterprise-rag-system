import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation" / "results"
DATASET = ROOT / "evaluation" / "dataset_financebench.json"

path = Path(sys.argv[1]) if len(sys.argv) > 1 else max(RESULTS.glob("run_*.json"), key=lambda p: p.stat().st_mtime)
cases = {c["id"]: c for c in json.load(open(DATASET, encoding="utf-8"))}
runs = json.load(open(path, encoding="utf-8"))
print("Scoring:", path.name, "\n")

# overall_status values produced by backend/app/answer.py
STATUS_MAP = {
    "supported": "verified",
    "needs_review": "rejected_by_verifier",
    "insufficient_evidence": "refused",
}


def nums(text):
    return {n.replace(",", "").rstrip(".") for n in re.findall(r"\d[\d,]*\.?\d*", str(text))}


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"


rows = []
for run in runs:
    c, r = cases[run["id"]], run["result"]
    chunks = r.get("retrieved_chunks") or []
    claims = r.get("claims") or []
    cits = [x for x in (r.get("citations") or []) if isinstance(x, dict)]
    pages = c.get("evidence_pages", [])
    status = "error" if r.get("error") else STATUS_MAP.get(r.get("overall_status"), "other")
    gen_text = " ".join(cl.get("claim", "") for cl in claims)  # kept even when the verifier rejects the answer
    ref_nums = nums(c.get("reference_answer", ""))
    num_ok = None
    if gen_text and ref_nums:
        num_ok = len(ref_nums & nums(gen_text)) / len(ref_nums) >= 0.5
    rows.append({
        "id": c["id"], "category": c["category"], "tier": c.get("tier", ""),
        "status": status, "overall_status": r.get("overall_status"),
        "source_hit": any(ch.get("source") == c["source"] for ch in chunks),
        "page_hit": any(ch.get("source") == c["source"] and ch.get("page") in pages for ch in chunks),
        "generated": bool(gen_text), "numeric_ok": num_ok,
        "n_claims": len(claims),
        "n_claims_supported": sum(cl.get("status") == "supported" for cl in claims),
        "any_contradicted": any(cl.get("status") == "contradicted" for cl in claims),
        "n_citations": len(cits),
        "n_citations_correct": sum(x.get("source") == c["source"] and x.get("page") in pages for x in cits),
        "latency_ms": r.get("latency_ms"), "error": r.get("error") or "",
    })

print("Pipeline statuses:", dict(Counter(r["overall_status"] for r in rows)))
ok = [r for r in rows if r["status"] != "error"]
print("Errors (excluded below):", len(rows) - len(ok), "\n")

for tier in ("extraction", "reasoning"):
    s = [r for r in ok if r["category"] == "supported" and r["tier"] == tier]
    if not s:
        continue
    gen = [r for r in s if r["generated"]]
    graded = [r for r in gen if r["numeric_ok"] is not None]
    print(f"SUPPORTED / {tier}  (n={len(s)})")
    print("  right document in context  :", pct(sum(r["source_hit"] for r in s), len(s)))
    print("  right page in context      :", pct(sum(r["page_hit"] for r in s), len(s)))
    print("  LLM produced an answer     :", pct(len(gen), len(s)))
    print("  numbers match reference    :", pct(sum(r["numeric_ok"] for r in graded), len(graded)), "(only questions with numbers)")
    print("  passed verification        :", pct(sum(r["status"] == "verified" for r in s), len(s)))
    print("  rejected by verifier       :", pct(sum(r["status"] == "rejected_by_verifier" for r in s), len(s)))
    print("  refused (no answer)        :", pct(sum(r["status"] == "refused" for r in s), len(s)))

u = [r for r in ok if r["category"] == "unsupported"]
print(f"\nUNSUPPORTED  (n={len(u)})")
print("  correctly refused          :", pct(sum(r["status"] == "refused" for r in u), len(u)))

k = [r for r in ok if r["category"] == "contradictory"]
print(f"\nCONTRADICTORY  (n={len(k)})")
print("  a claim flagged contradicted:", pct(sum(r["any_contradicted"] for r in k), len(k)))
print("  refused                     :", pct(sum(r["status"] == "refused" for r in k), len(k)))

v = [r for r in ok if r["category"] == "supported" and r["status"] == "verified"]
tot_c = sum(r["n_citations"] for r in v)
tot_cl = sum(r["n_claims"] for r in ok)
print("\nCITATIONS / CLAIMS")
print("  citations on a gold evidence page (verified answers):", pct(sum(r["n_citations_correct"] for r in v), tot_c))
print("  claims judged supported by the verifier             :", pct(sum(r["n_claims_supported"] for r in ok), tot_cl))

csv_path = RESULTS / f"per_case_{path.stem}.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\nPer-case table:", csv_path)