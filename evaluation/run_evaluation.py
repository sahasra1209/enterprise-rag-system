import argparse
import contextlib
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app"))

from answer import generate_answer

DATASET = ROOT / "evaluation" / "dataset_financebench.json"
RESULTS = ROOT / "evaluation" / "results"

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, help="run only the first N cases")
ap.add_argument("--category", help="supported / unsupported / contradictory")
ap.add_argument("--out", help="results file (re-use a name to resume a run)")
args = ap.parse_args()

cases = json.load(open(DATASET, encoding="utf-8"))
if args.category:
    cases = [c for c in cases if c["category"] == args.category]
if args.limit:
    cases = cases[: args.limit]

RESULTS.mkdir(exist_ok=True)
out = Path(args.out) if args.out else RESULTS / f"run_{datetime.now():%Y%m%d_%H%M%S}.json"

done = {}
if out.exists():
    done = {r["id"]: r for r in json.load(open(out, encoding="utf-8"))}
    print(f"Resuming: {len(done)} cases already in {out.name}")


def save():
    rows = [done[c["id"]] for c in cases if c["id"] in done]
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


for i, c in enumerate(cases, 1):
    if c["id"] in done:
        continue
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):  # hide the pipeline's debug prints
            r = generate_answer(c["question"])
    except Exception as e:
        r = {"error": f"{type(e).__name__}: {e}"}
    done[c["id"]] = {"id": c["id"], "wall_s": round(time.time() - t0, 1), "result": r}
    save()
    print(f"[{i}/{len(cases)}] {c['id']} -> {r.get('overall_status')} ({done[c['id']]['wall_s']}s)")

print("Saved:", out)