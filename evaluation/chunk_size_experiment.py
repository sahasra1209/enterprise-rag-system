import json
import re
import sys
from pathlib import Path
from statistics import mean

import chromadb
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app"))

from ingestion import extract_pages_from_pdf
from chunking import split_text_into_chunks
from embeddings import load_embedding_model

PDF_DIR = Path(r"E:\kg25\RAG-Datasets\financebench-main\pdfs")
DATASET = ROOT / "evaluation" / "dataset_financebench.json"
OUT = ROOT / "evaluation" / "results" / "chunk_size_experiment.json"

DOCS = ["AMD_2022_10K", "AMERICANEXPRESS_2022_10K", "BOEING_2022_10K", "AMCOR_2023_10K"]
CONFIGS = [(200, "policy_documents"), (600, "eval_c600"), (1000, "eval_c1000")]
BATCH = 4000
N = 10

client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
model = load_embedding_model()
print("Embedding model max_seq_length:", getattr(model, "max_seq_length", "unknown"))


def build(size, name):
    col = client.get_or_create_collection(name=name)
    for doc in DOCS:
        source = f"{doc}.pdf"
        if col.get(where={"source": source}, limit=1, include=[])["ids"]:
            continue  # already indexed (includes the team's 200-char collection)
        print(f"  indexing {source} at {size} chars ...")
        chunks = []
        for page in extract_pages_from_pdf(str(PDF_DIR / source)):
            for n, text in enumerate(
                split_text_into_chunks(page["text"], chunk_size=size, overlap=size // 5), 1
            ):
                chunks.append((source, page["page_number"], n, text))
        embs = model.encode([c[3] for c in chunks])
        for i in range(0, len(chunks), BATCH):
            part = chunks[i:i + BATCH]
            col.upsert(
                ids=[f"{s}|{p}|{n}" for s, p, n, _ in part],
                documents=[t for _, _, _, t in part],
                embeddings=[e.tolist() for e in embs[i:i + BATCH]],
                metadatas=[{"source": s, "page_number": p, "chunk_number": n} for s, p, n, _ in part],
            )
    return col


def numbers(text):
    return {t.replace(",", "").rstrip(".") for t in re.findall(r"\d[\d,]*\.?\d*", text)}


def key_numbers(text):
    return {t for t in numbers(text)
            if len(re.sub(r"\D", "", t)) >= 3 and not (t.isdigit() and 1990 <= int(t) <= 2030)}


cols = {}
for size, name in CONFIGS:
    print(f"Preparing {size}-char chunks ({name})")
    cols[name] = build(size, name)

cases = [c for c in json.load(open(DATASET, encoding="utf-8"))
         if c["category"] == "supported" and c["source"][:-4] in DOCS]
print(f"\nQuestions: {len(cases)}")

rows = []
for size, name in CONFIGS:
    for c in cases:
        res = cols[name].query(
            query_embeddings=model.encode([c["question"]]).tolist(),
            n_results=N, where={"source": c["source"]}, include=["documents", "metadatas"],
        )
        docs, metas = res["documents"][0], res["metadatas"][0]
        rank = next((i for i, m in enumerate(metas, 1) if m["page_number"] in c["evidence_pages"]), None)
        gold = key_numbers(" ".join(c["evidence_text"]))
        cov = len(gold & numbers(" ".join(docs[:3]))) / len(gold) if len(gold) >= 3 else None
        rows.append({"size": size, "id": c["id"], "tier": c["tier"], "rank": rank,
                     "coverage3": cov, "context_chars_top3": sum(len(d) for d in docs[:3])})

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def summarize(size, tier=None):
    r = [x for x in rows if x["size"] == size and (tier is None or x["tier"] == tier)]
    if not r:
        return
    rec = lambda k: sum(1 for x in r if x["rank"] and x["rank"] <= k) / len(r)
    mrr = sum(1 / x["rank"] for x in r if x["rank"]) / len(r)
    cov = [x["coverage3"] for x in r if x["coverage3"] is not None]
    print(f"  {size:>5} chars | page recall @1={rec(1):.0%} @3={rec(3):.0%} @10={rec(10):.0%} | MRR={mrr:.2f}"
          f" | gold-number coverage in top-3: {mean(cov):.0%} (n={len(cov)})"
          f" | avg top-3 context: {mean(x['context_chars_top3'] for x in r):.0f} chars")


for label, tier in [("ALL", None), ("EXTRACTION", "extraction"), ("REASONING", "reasoning")]:
    print(f"\n=== {label} ===")
    for size, _ in CONFIGS:
        summarize(size, tier)

print("\n=== Hit@3 vs the 200-char baseline (same questions) ===")
base = {x["id"]: x["rank"] for x in rows if x["size"] == 200}
for size, _ in CONFIGS[1:]:
    cur = {x["id"]: x["rank"] for x in rows if x["size"] == size}
    hit = lambda r: bool(r and r <= 3)
    gained = sum(1 for i in base if not hit(base[i]) and hit(cur[i]))
    lost = sum(1 for i in base if hit(base[i]) and not hit(cur[i]))
    p = binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else float("nan")
    print(f"  {size} chars: gained={gained} lost={lost} sign-test p={p:.3f}")
print("\nSaved:", OUT)