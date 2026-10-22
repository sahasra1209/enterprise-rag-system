
from pathlib import Path

from ingestion import extract_pages_from_pdf
from chunking import split_text_into_chunks
from embeddings import load_embedding_model
from vector_store import create_vector_store, store_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def ingest_all_pdfs():
    """Ingest every PDF in data/raw into ChromaDB."""

    pdf_files = sorted(RAW_DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {RAW_DATA_DIR}")
        return

    all_chunks = []

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")

        pages = extract_pages_from_pdf(str(pdf_path))
        source_chunks = []

        for page in pages:
            chunks = split_text_into_chunks(
                page["text"],
                chunk_size=500,
                overlap=100
            )

            for chunk_number, chunk_text in enumerate(
                chunks, start=1
            ):
                source_chunks.append({
                    "source": pdf_path.name,
                    "page_number": page["page_number"],
                    "chunk_number": chunk_number,
                    "text": chunk_text
                })

        print(f"Pages extracted: {len(pages)}")
        print(f"Chunks created: {len(source_chunks)}")

        all_chunks.extend(source_chunks)

    if not all_chunks:
        print("No text chunks were extracted from the PDFs.")
        return

    model = load_embedding_model()

    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = model.encode(texts)

    print(f"\nPDF files processed: {len(pdf_files)}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Total embeddings: {len(embeddings)}")

    collection = create_vector_store()
    store_chunks(collection, all_chunks, embeddings)

    print("\nMulti-PDF ingestion completed successfully!")


if __name__ == "__main__":
    ingest_all_pdfs()
