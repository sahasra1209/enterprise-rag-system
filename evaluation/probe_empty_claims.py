import contextlib
import io
import json
import re
import sys
from pathlib import Path

import chromadb
import ollama as real_ollama

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app"))

import answer
from embeddings import load_embedding_model

src = (ROOT / "evaluation" / "run_prompt_c.py").read_text(encoding="utf-8")
PROMPT = re.search(r'PROMPT_B = """(.*?)"""', src, re.S).group(1).strip()
IDS = ["financebench_id_01279", "financebench_id_00563", "financebench_id_00517", "financebench_id_00585"]
raw_log = []


class Logged:
    @staticmethod
    def chat(model, messages, options=None, **kw):
        m = re.search(r"Evidence:\n(.*)\n\nQuestion:\n(.*)\n\nAnswer:", messages[0]["content"], re.S)
        if m:
            messages = [{"role": "user", "content": PROMPT.format(
                fallback=answer.FALLBACK, context=m.group(1), question=m.group(2).strip())}]
        resp = real_ollama.chat(model=model, messages=messages, options=options, **kw)
        raw_log.append(resp["message"]["content"])
        return resp


answer.ollama = Logged
embedder = load_embedding_model()
col = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection("eval_c1000")
reranker = answer.get_reranker()
cases = {c["id"]: c for c in json.load(open(ROOT / "evaluation" / "dataset_financebench.json", encoding="utf-8"))}

for cid in IDS:
    c = cases[cid]
    res = col.query(query_embeddings=embedder.encode([c["question"]]).tolist(), n_results=20,
                    where={"source": c["source"]}, include=["documents", "metadatas"])
    ranked = reranker.rerank(c["question"], res["documents"][0], res["metadatas"][0], top_k=3)
    docs = [r["text"] for r in ranked]
    metas = [r["metadata"] for r in ranked]
    answer.search_documents = lambda q, top_k=3, *a, _d=docs, _m=metas, **k: {"documents": [_d], "metadatas": [_m]}
    with contextlib.redirect_stdout(io.StringIO()):
        r = answer.generate_answer(c["question"])
    print("=" * 70)
    print(cid, "|", c["question"][:110])
    print("REFERENCE:", c["reference_answer"][:150].replace("\n", " "))
    print("STATUS   :", r["overall_status"])
    print("RAW LLM OUTPUT:", repr(raw_log[-1][:600]))