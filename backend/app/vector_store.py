
import hashlib
import chromadb


def create_vector_store():
    """Create or load a persistent ChromaDB collection."""

    client = chromadb.PersistentClient(path="./chroma_db")

    collection = client.get_or_create_collection(
        name="policy_documents"
    )

    return collection


def store_chunks(collection, chunks, embeddings):
    """Store chunks using stable, source-specific IDs."""

    if len(chunks) != len(embeddings):
        raise ValueError(
            "The number of chunks must match the number of embeddings."
        )

    if not chunks:
        print("No chunks to store.")
        return

    # Refresh each source so re-ingestion doesn't leave stale chunks.
    sources = {chunk["source"] for chunk in chunks}

    for source in sources:
        existing = collection.get(
            where={"source": source},
            include=[]
        )
        existing_ids = existing.get("ids", [])

        if existing_ids:
            collection.delete(ids=existing_ids)

    ids = []
    documents = []
    vectors = []
    metadatas = []

    for chunk, embedding in zip(chunks, embeddings):
        source = chunk["source"]
        page_number = chunk["page_number"]
        chunk_number = chunk["chunk_number"]

        # Stable ID: same source/page/chunk gets the same ID.
        identifier = (
            f"{source}|{page_number}|{chunk_number}|{chunk['text']}"
        )
        chunk_id = hashlib.sha256(
            identifier.encode("utf-8")
        ).hexdigest()

        ids.append(chunk_id)
        documents.append(chunk["text"])

        vectors.append(
            embedding.tolist()
            if hasattr(embedding, "tolist")
            else list(embedding)
        )

        metadatas.append({
            "source": source,
            "page_number": page_number,
            "chunk_number": chunk_number
        })

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=vectors,
        metadatas=metadatas
    )

    print(f"Stored {len(chunks)} chunks successfully.")
    print(f"Sources refreshed: {len(sources)}")
    print(f"Total chunks in collection: {collection.count()}")
