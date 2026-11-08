
import contextlib
import io
import json
import re
import sys
import time
from pathlib import Path

import chromadb
import ollama as real_ollama

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app"))

import answer
from embeddings import load_embedding_model

DATASET = ROOT / "evaluation" / "dataset_financebench.json"
OUT = ROOT / "evaluation" / "results" / "run_prompt_d.json"

DOCS = [
    "AMD_2022_10K",
    "AMERICANEXPRESS_2022_10K",
    "BOEING_2022_10K",
    "AMCOR_2023_10K"
]

COLLECTION = "eval_c1000"
POOL = 20


PROMPT_B = """
You answer questions about company annual reports (10-K filings).

Use ONLY the evidence below. Never reply with only an evidence ID. First write the answer as a sentence, then add the evidence ID. Answer in at most three short sentences. Do not comment on the evidence or explain where the information comes from.

Rules:
- Every sentence must end with an evidence ID such as [E1], placed before the final period.
- Use only the evidence IDs shown below.
- Numbers and ratios may be computed from figures in the evidence.
- If the evidence truly contains nothing relevant, reply exactly: {fallback}
- Never reply with only an evidence ID. First write the answer as a sentence, then add the evidence ID.

Evidence:
{context}

Question:
{question}

Answer:
""".strip()


class PatchedOllama:
    """Swaps the pipeline's prompt for PROMPT_B at run time; no team file is edited."""

    @staticmethod
    def chat(model, messages, options=None, **kw):
        m = re.search(
            r"Evidence:\n(.*)\n\nQuestion:\n(.*)\n\nAnswer:",
            messages[0]["content"],
            re.S
        )

        if m:
            prompt = PROMPT_B.format(
                fallback=answer.FALLBACK,
                context=m.group(1),
                question=m.group(2).strip()
            )

            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

        return real_ollama.chat(
            model=model,
            messages=messages,
            options=options,
            **kw
        )


answer.ollama = PatchedOllama

embedder = load_embedding_model()

col = chromadb.PersistentClient(
    path=str(ROOT / "chroma_db")
).get_collection(COLLECTION)

reranker = answer.get_reranker()


cases = [
    c for c in json.load(open(DATASET, encoding="utf-8"))
    if c["category"] in ("supported", "unsupported")
    and c["source"][:-4] in DOCS
]

print(f"Cases: {len(cases)}")


rows = []

for i, c in enumerate(cases, 1):

    emb = embedder.encode([c["question"]]).tolist()

    res = col.query(
        query_embeddings=emb,
        n_results=POOL,
        where={"source": c["source"]},
        include=["documents", "metadatas"]
    )

    ranked = reranker.rerank(
        c["question"],
        res["documents"][0],
        res["metadatas"][0],
        top_k=3
    )

    top_docs = [r["text"] for r in ranked]
    top_metas = [r["metadata"] for r in ranked]

    answer.search_documents = (
        lambda q, top_k=3, *a, _d=top_docs, _m=top_metas, **k: {
            "documents": [_d],
            "metadatas": [_m]
        }
    )

    t0 = time.time()

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = answer.generate_answer(c["question"])

    except Exception as e:
        r = {
            "error": f"{type(e).__name__}: {e}"
        }

    rows.append({
        "id": c["id"],
        "wall_s": round(time.time() - t0, 1),
        "result": r
    })

    print(
        f"[{i}/{len(cases)}] "
        f"{c['id']} -> "
        f"{r.get('overall_status')} "
        f"({rows[-1]['wall_s']}s)"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)

    OUT.write_text(
        json.dumps(
            rows,
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )


print("\nSaved:", OUT)
print(f"Total evaluated cases: {len(rows)}")
