import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "evaluation" / "results" / "run_best_config.json"
cases = {c["id"]: c for c in json.load(open(ROOT / "evaluation" / "dataset_financebench.json", encoding="utf-8"))}
runs = json.load(open(path, encoding="utf-8"))

rows, claim_status, ver_status, details = [], Counter(), Counter(), []

for run in runs:
    c = cases[run["id"]]
    if c["category"] != "supported":
        continue
    r = run["result"]
    chunks = r.get("retrieved_chunks") or []
    claims = r.get("claims") or []
    pages = c.get("evidence_pages", [])
    page_hit = any(ch.get("source") == c["source"] and ch.get("page") in pages for ch in chunks)
    answered = bool(claims)

    if r.get("overall_status") == "supported":
        stage = "verified"
    elif answered:
        stage = "answered, rejected by verifier" + (" (evidence page in context)" if page_hit else " (evidence page NOT in context)")
    else:
        stage = "refused, evidence page in context" if page_hit else "refused, evidence page NOT in context"

    for cl in claims:
        claim_status[cl.get("status")] += 1
        for v in cl.get("verification_results") or []:
            ver_status[v.get("status")] += 1

    if answered:
        details.append((c, page_hit, claims))
    rows.append({
        "id": c["id"], "tier": c["tier"], "stage": stage, "page_hit": page_hit,
        "question": c["question"], "reference": c["reference_answer"],
        "generated": " | ".join(cl.get("claim", "") for cl in claims),
        "your_label": "", "your_notes": "",
    })

print(f"Supported questions analysed: {len(rows)}\n")
print("=== Where each question ended up ===")
for stage, n in sorted(Counter(r["stage"] for r in rows).items(), key=lambda x: -x[1]):
    print(f"  {n:>2}  {stage}")

print("\n=== Claim statuses (all generated claims) ===")
for s, n in claim_status.most_common():
    print(f"  {n:>2}  {s}")
print("\n=== Verifier verdicts per (claim, cited passage) ===")
for s, n in ver_status.most_common():
    print(f"  {n:>2}  {s}")

print("\n=== Every answered question ===")
for c, page_hit, claims in details:
    print("\n" + "-" * 70)
    print(c["id"], "| evidence page in context:", page_hit)
    print("Q  :", c["question"][:140])
    print("REF:", c["reference_answer"][:200].replace("\n", " "))
    for cl in claims:
        print(f"  [{cl.get('status')}] {cl.get('claim', '')[:200]}")

out = ROOT / "evaluation" / "results" / "error_analysis.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\nSaved:", out)