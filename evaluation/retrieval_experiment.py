import json
import sys
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app"))

from embeddings import load_embedding_model
from reranker import DocumentReranker

DATASET = ROOT / "evaluation" / "dataset_financebench.json"
OUT = ROOT / "evaluation" / "results" / "retrieval_experiment.json"
N = 50
RERANK_POOLS = [20, 50]

cases = [c for c in json.load(open(DATASET, encoding="utf-8")) if c["category"] == "supported"]
col = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection("policy_documents")
model = load_embedding_model()
reranker = DocumentReranker()


def is_hit(meta, case):
    return meta.get("source") == case["source"] and meta.get("page_number") in case["evidence_pages"]


def first_rank(metas, case):
    for i, m in enumerate(metas, 1):
        if is_hit(m, case):
            return i
    return None


rows = []
for n, c in enumerate(cases, 1):
    q = c["question"]
    emb = model.encode([q]).tolist()
    res = col.query(query_embeddings=emb, n_results=N, include=["documents", "metadatas"])
    docs, metas = res["documents"][0], res["metadatas"][0]

    row = {"id": c["id"], "tier": c["tier"], "vector": first_rank(metas, c)}
    for pool in RERANK_POOLS:
        ranked = reranker.rerank(q, docs[:pool], metas[:pool], top_k=pool)
        row[f"rerank{pool}"] = first_rank([x["metadata"] for x in ranked], c)
    rows.append(row)
    print(f"[{n}/{len(cases)}] {c['id']}  vector={row['vector']}  rerank20={row['rerank20']}  rerank50={row['rerank50']}")

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def recall(method, k, subset):
    return sum(1 for r in subset if r[method] is not None and r[method] <= k) / len(subset)


def mrr(method, subset):
    return sum(1 / r[method] for r in subset if r[method]) / len(subset)


print("\n=== Page-level recall (right page among top K) ===")
for label, subset in [("ALL", rows)] + [(t, [r for r in rows if r["tier"] == t]) for t in ("extraction", "reasoning")]:
    print(f"\n{label} (n={len(subset)})")
    print(f"  vector only        : " + "  ".join(f"@{k}={recall('vector', k, subset):.0%}" for k in (3, 5, 10, 20, 50)) + f"  MRR={mrr('vector', subset):.2f}")
    for pool in RERANK_POOLS:
        m = f"rerank{pool}"
        print(f"  rerank top-{pool:<2} pool : " + "  ".join(f"@{k}={recall(m, k, subset):.0%}" for k in (3, 5, 10)) + f"  MRR={mrr(m, subset):.2f}")
print("\nSaved:", OUT)