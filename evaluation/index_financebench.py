import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app"))

from ingestion import extract_pages_from_pdf
from chunking import split_text_into_chunks
from embeddings import load_embedding_model
from vector_store import create_vector_store

PDF_DIR = Path(r"E:\kg25\RAG-Datasets\financebench-main\pdfs")

DOCS = [
    "AMD_2022_10K",
    "AMERICANEXPRESS_2022_10K",
    "BOEING_2022_10K",
    "PEPSICO_2022_10K",
    "AMCOR_2023_10K",
    "3M_2022_10K",
    "BESTBUY_2023_10K",
    "CVSHEALTH_2022_10K",
]

CHUNK_SIZE = 200
OVERLAP = 40
BATCH = 4000  # ChromaDB's limit is 5461 records per call


def make_id(source, page, chunk_no, text):
    # Same ID scheme as backend/app/vector_store.py
    return hashlib.sha256(f"{source}|{page}|{chunk_no}|{text}".encode("utf-8")).hexdigest()


def is_indexed(collection, source):
    return bool(collection.get(where={"source": source}, limit=1, include=[])["ids"])


def store_in_batches(collection, chunks, embeddings):
    for start in range(0, len(chunks), BATCH):
        part = chunks[start:start + BATCH]
        vecs = embeddings[start:start + BATCH]
        collection.upsert(
            ids=[make_id(c["source"], c["page_number"], c["chunk_number"], c["text"]) for c in part],
            documents=[c["text"] for c in part],
            embeddings=[v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs],
            metadatas=[
                {"source": c["source"], "page_number": c["page_number"], "chunk_number": c["chunk_number"]}
                for c in part
            ],
        )


model = load_embedding_model()
collection = create_vector_store()

for doc in DOCS:
    source = f"{doc}.pdf"

    if is_indexed(collection, source):
        print(f"SKIP (already indexed): {source}")
        continue

    pdf_path = PDF_DIR / source
    if not pdf_path.exists():
        print(f"SKIP (file not found): {pdf_path}")
        continue

    print(f"Indexing {doc} ...")
    pages = extract_pages_from_pdf(str(pdf_path))
    chunks = []
    for page in pages:
        for n, text in enumerate(
            split_text_into_chunks(page["text"], chunk_size=CHUNK_SIZE, overlap=OVERLAP),
            start=1,
        ):
            chunks.append({
                "source": source,
                "page_number": page["page_number"],
                "chunk_number": n,
                "text": text,
            })

    if not chunks:
        print(f"  no text found in {doc}, skipping")
        continue

    embeddings = model.encode([c["text"] for c in chunks])
    store_in_batches(collection, chunks, embeddings)
    print(f"  {len(pages)} pages, {len(chunks)} chunks stored")

print("Total records in database:", collection.count())