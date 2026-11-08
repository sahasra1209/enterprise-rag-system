import contextlib
import io
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app"))

import answer  # the team's module; we swap one function at run time, no file is edited

SRC = Path(r"E:\kg25\RAG-Datasets\financebench-main\data\financebench_open_source.jsonl")
DATASET = ROOT / "evaluation" / "dataset_financebench.json"
OUT = ROOT / "evaluation" / "results" / "run_oracle.json"
PAGE_OFFSET = 1    # verified earlier: matched all 39 evidence items
MAX_PASSAGES = 2   # keep the prompt small for the 3B model
MAX_CHARS = 1800

raw = pd.read_json(SRC, lines=True).set_index("financebench_id")
cases = [c for c in json.load(open(DATASET, encoding="utf-8")) if c["category"] == "supported"]


def gold_search(docs, metas):
    def _search(question, top_k=3, *args, **kwargs):
        return {"documents": [docs], "metadatas": [metas]}
    return _search


rows = []
for i, c in enumerate(cases, 1):
    ev = list(raw.loc[c["id"], "evidence"])[:MAX_PASSAGES]
    docs = [e["evidence_text"][:MAX_CHARS] for e in ev]
    metas = [{"source": f"{e['doc_name']}.pdf", "page_number": int(e["evidence_page_num"]) + PAGE_OFFSET} for e in ev]
    answer.search_documents = gold_search(docs, metas)
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = answer.generate_answer(c["question"])
    except Exception as e:
        r = {"error": f"{type(e).__name__}: {e}"}
    rows.append({"id": c["id"], "wall_s": round(time.time() - t0, 1), "result": r})
    print(f"[{i}/{len(cases)}] {c['id']} -> {r.get('overall_status')} ({rows[-1]['wall_s']}s)")
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

print("Saved:", OUT)