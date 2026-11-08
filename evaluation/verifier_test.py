import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app"))

from hallucination_detector import HallucinationDetector

DATASET = ROOT / "evaluation" / "dataset_financebench.json"
OUT = ROOT / "evaluation" / "results" / "verifier_test.json"
WINDOW = 150  # characters of context on each side of the sentence

NUM = re.compile(r"\d[\d,]*\.?\d*")


def pick_sentence(text):
    """First sentence with 8-40 words, mostly words (not a table row), containing a 2+ digit number."""
    flat = " ".join(text.split())
    pos = 0
    for s in re.split(r"(?<=[.!?])\s+", flat):
        start = flat.find(s, pos)
        if start < 0:
            continue
        end = start + len(s)
        pos = end
        words = s.split()
        if not 8 <= len(words) <= 40:
            continue
        if sum(w.strip(".,;:()$%").isalpha() for w in words) / len(words) < 0.6:
            continue
        if not any(len(re.sub(r"\D", "", m)) >= 2 for m in NUM.findall(s)):
            continue
        return s, flat[max(0, start - WINDOW): end + WINDOW]
    return None


def alter_number(sentence):
    for m in NUM.findall(sentence):
        if len(re.sub(r"\D", "", m)) < 2:
            continue
        first = next(ch for ch in m if ch.isdigit())
        choices = [d for d in "123456789" if d != first]
        new_first = choices[len(sentence) % len(choices)]
        new = m.replace(first, new_first, 1)
        return sentence.replace(m, new, 1), m, new
    return None


cases = [c for c in json.load(open(DATASET, encoding="utf-8")) if c["category"] == "supported"]
items = []
for c in cases:
    for text in c["evidence_text"]:
        picked = pick_sentence(text)
        if picked:
            items.append({"id": c["id"], "source": c["source"], "claim": picked[0], "passage": picked[1]})
            break
print(f"Usable cases: {len(items)} of {len(cases)}")

det = HallucinationDetector()
rows = []


def check(kind, claim, passage, item, extra=None):
    try:
        res = det.verify_claim(claim, passage)
        status = res.get("status")
        ent = (res.get("scores") or {}).get("entailment")
    except Exception as e:
        status, ent = f"error: {type(e).__name__}", None
    rows.append({"kind": kind, "id": item["id"], "claim": claim, "status": status, "entailment": ent, **(extra or {})})


for i, it in enumerate(items):
    check("verbatim", it["claim"], it["passage"], it)

    alt = alter_number(it["claim"])
    if alt:
        check("altered_number", alt[0], it["passage"], it, {"changed": f"{alt[1]} -> {alt[2]}"})

    other = next((o for o in items[i + 1:] + items[:i] if o["source"] != it["source"]), None)
    if other:
        check("unrelated", other["claim"], it["passage"], it)

OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n=== Verifier results by claim type ===")
for kind in ("verbatim", "altered_number", "unrelated"):
    sub = [r for r in rows if r["kind"] == kind]
    cnt = Counter(r["status"] for r in sub)
    print(f"\n{kind} (n={len(sub)}): {dict(cnt)}")

verb = [r for r in rows if r["kind"] == "verbatim" and r["status"] != "supported"]
alt = [r for r in rows if r["kind"] == "altered_number" and r["status"] == "supported"]
print("\nVerbatim claims NOT accepted (first 3):")
for r in verb[:3]:
    print(" -", r["status"], "|", r["claim"][:140])
print("\nAltered claims wrongly accepted as supported (first 3):")
for r in alt[:3]:
    print(" -", r["changed"], "|", r["claim"][:140])
print("\nSaved:", OUT)