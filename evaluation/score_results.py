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


def is_refusal(r):
    return r.get("overall_status") == "insufficient_evidence" or "couldn't find a supported answer" in str(r.get("answer", "")).lower()


def nums(text):
    return {n.replace(",", "").rstrip(".") for n in re.findall(r"\d[\d,]*\.?\d*", str(text))}


def citation_hits(r, case):
    """Return (n_citations, n_correct). Tolerant of unknown citation format."""
    chunks = {c.get("citation_id"): c for c in r.get("retrieved_chunks", [])}
    good = total = 0
    for cit in r.get("citations") or []:
        if isinstance(cit, str):
            cit = chunks.get(cit, {"citation_id": cit})
        elif isinstance(cit, dict) and "citation_id" in cit and "source" not in cit:
            cit = {**chunks.get(cit["citation_id"], {}), **cit}
        if not isinstance(cit, dict):
            continue
        total += 1
        src, page = cit.get("source"), cit.get("page", cit.get("page_number"))
        if src == case["source"] and page in case.get("evidence_pages", []):
            good += 1
    return total, good


rows = []
for run in runs:
    c, r = cases[run["id"]], run["result"]
    chunks = r.get("retrieved_chunks", []) or []
    status = "error" if r.get("error") else ("refused" if is_refusal(r) else "answered")
    src_hit = any(ch.get("source") == c["source"] for ch in chunks)
    page_hit = any(ch.get("source") == c["source"] and ch.get("page") in c.get("evidence_pages", []) for ch in chunks)
    ref_nums = nums(c.get("reference_answer", ""))
    num_ok = None
    if status == "answered" and ref_nums:
        num_ok = len(ref_nums & nums(r.get("answer", ""))) / len(ref_nums) >= 0.5
    n_cit, n_good = citation_hits(r, c) if status == "answered" else (0, 0)
    rows.append({
        "id": c["id"], "category": c["category"], "tier": c.get("tier", ""), "status": status,
        "overall_status": r.get("overall_status"), "source_hit": src_hit, "page_hit": page_hit,
        "numeric_ok": num_ok, "n_citations": n_cit, "n_citations_correct": n_good,
        "latency_ms": r.get("latency_ms"), "error": r.get("error") or "",
    })


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"


print("Pipeline statuses:", dict(Counter(r["overall_status"] for r in rows)))
print("Errors (excluded below):", sum(r["status"] == "error" for r in rows), "\n")
ok = [r for r in rows if r["status"] != "error"]

for tier in ("extraction", "reasoning"):
    s = [r for r in ok if r["category"] == "supported" and r["tier"] == tier]
    if not s:
        continue
    ans = [r for r in s if r["status"] == "answered"]
    print(f"SUPPORTED / {tier}  (n={len(s)})")
    print("  right document retrieved :", pct(sum(r["source_hit"] for r in s), len(s)))
    print("  right PAGE retrieved     :", pct(sum(r["page_hit"] for r in s), len(s)))
    print("  answered (not refused)   :", pct(len(ans), len(s)))
    graded = [r for r in ans if r["numeric_ok"] is not None]
    print("  numbers match (rough)    :", pct(sum(r["numeric_ok"] for r in graded), len(graded)))

u = [r for r in ok if r["category"] == "unsupported"]
print(f"\nUNSUPPORTED  (n={len(u)})\n  correctly refused        :", pct(sum(r["status"] == "refused" for r in u), len(u)))

k = [r for r in ok if r["category"] == "contradictory"]
flagged = [r for r in k if "contradict" in str(r["overall_status"]).lower()]
print(f"\nCONTRADICTORY  (n={len(k)})\n  flagged as contradicted  :", pct(len(flagged), len(k)))

ans_all = [r for r in ok if r["status"] == "answered" and r["category"] == "supported"]
tot = sum(r["n_citations"] for r in ans_all)
print("\nCITATIONS\n  correct citations        :", pct(sum(r["n_citations_correct"] for r in ans_all), tot) if tot else "n/a (no citations returned)")

csv_path = RESULTS / f"per_case_{path.stem}.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\nPer-case table:", csv_path)