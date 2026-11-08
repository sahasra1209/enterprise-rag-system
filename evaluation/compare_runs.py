import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "evaluation" / "results"
cases = {c["id"]: c for c in json.load(open(ROOT / "evaluation" / "dataset_financebench.json", encoding="utf-8"))}

RUNS = [
    ("Pipeline as shipped (200-char chunks, top-3)", "run_baseline.json"),
    ("Prompt B (1,000-char chunks, pool 20, rerank 3)", "run_best_config.json"),
    ("Prompt C", "run_prompt_c.json"),
    ("Prompt D", "run_prompt_d.json"),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, CI {100 * lo:.0f}-{100 * hi:.0f}%)"


loaded = {}
for label, fname in RUNS:
    rows = json.load(open(R / fname, encoding="utf-8"))
    loaded[label] = {r["id"]: r["result"] for r in rows}

sup_ids = sorted(set.intersection(*[{i for i in d if cases[i]["category"] == "supported"} for d in loaded.values()]))
uns_ids = sorted(set.intersection(*[{i for i in d if cases[i]["category"] == "unsupported"} for d in loaded.values()]))
print(f"Supported questions common to all runs: {len(sup_ids)}")
print(f"Unsupported questions common to all runs: {len(uns_ids)} {uns_ids}\n")


def has_text(r):
    return any((cl.get("claim") or "").strip() for cl in (r.get("claims") or []))


def metrics(res, ids):
    m = dict(page=0, refused=0, empty=0, text=0, verified=0, rejected_text=0, errors=0)
    for i in ids:
        r = res[i]
        pages = cases[i].get("evidence_pages", [])
        if r.get("error"):
            m["errors"] += 1
        if any(c.get("source") == cases[i]["source"] and c.get("page") in pages for c in (r.get("retrieved_chunks") or [])):
            m["page"] += 1
        st = r.get("overall_status")
        if st == "insufficient_evidence":
            m["refused"] += 1
        elif st == "supported":
            m["verified"] += 1
            m["text"] += 1
        elif r.get("claims"):
            if has_text(r):
                m["text"] += 1
                m["rejected_text"] += 1
            else:
                m["empty"] += 1
    return m


n = len(sup_ids)
lines = [f"| Run (n={n} supported questions) | Gold page in context | Refused | Empty reply (tag only) | Answer with real text | Verified | Real text, rejected |",
         "|---|---|---|---|---|---|---|"]
for label, _ in RUNS:
    m = metrics(loaded[label], sup_ids)
    lines.append(f"| {label} | {fmt(m['page'], n)} | {fmt(m['refused'], n)} | {fmt(m['empty'], n)} | "
                 f"{fmt(m['text'], n)} | {fmt(m['verified'], n)} | {fmt(m['rejected_text'], n)} |")
    if m["errors"]:
        print(f"WARNING: {label} has {m['errors']} cases with an error field")

u = len(uns_ids)
lines += ["", f"| Run (unsupported, n={u} common) | Refused | Answered |", "|---|---|---|"]
for label, _ in RUNS:
    k = sum(loaded[label][i].get("overall_status") == "insufficient_evidence" for i in uns_ids)
    lines.append(f"| {label} | {fmt(k, u)} | {u - k} |")

for label in ("Prompt C", "Prompt D"):
    own = [i for i in loaded[label] if cases[i]["category"] == "unsupported"]
    k = sum(loaded[label][i].get("overall_status") == "insufficient_evidence" for i in own)
    lines.append(f"\n{label}, all unsupported questions: refused {fmt(k, len(own))}")

first = R / "run_prompt_c_first.json"
if first.exists():
    a = {r["id"]: r["result"].get("overall_status") for r in json.load(open(first, encoding="utf-8"))}
    b = {i: loaded["Prompt C"][i].get("overall_status") for i in loaded["Prompt C"]}
    common = [i for i in a if i in b]
    same = sum(a[i] == b[i] for i in common)
    lines.append(f"\nRepeatability: prompt C run twice, same overall_status on {same}/{len(common)} shared questions")

text = "\n".join(lines)
print(text)
(R / "comparison.md").write_text(text, encoding="utf-8")
print("\nSaved:", R / "comparison.md")