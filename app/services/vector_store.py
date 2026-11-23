from pathlib import Path

import chromadb


# Persistent ChromaDB storage directory
CHROMA_DIR = Path(__file__).resolve().parents[2] / "storage" / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# Initialize ChromaDB client
client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


# Create or retrieve the collection
collection = client.get_or_create_collection(
    name="docmind_chunks",
    metadata={"hnsw:space": "cosine"}
)


def add_chunks(
    document_id: str,
    filename: str,
    total_pages: int,
    chunks: list[dict],
    embeddings: list[list[float]]
):

    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks and embeddings must match.")

    if not chunks:
        return

    ids = [
        f"{document_id}_{chunk['chunk_id']}"
        for chunk in chunks
    ]

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "document_id": document_id,
            "filename": filename,
            "total_pages": total_pages,
            "chunk_id": chunk["chunk_id"],
            "page_number": chunk["page_number"]
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )


def search_similar_chunks(
    query_embedding: list[float],
    top_k: int = 3,
    document_id: str | None = None
) -> list[dict]:

    if top_k < 1:
        raise ValueError("top_k must be greater than zero.")

    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    filters = None

    if document_id:
        filters = {"document_id": document_id}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, total_chunks),
        where=filters,
        include=["documents", "metadatas", "distances"]
    )

    matched_ids = results["ids"][0]

    if not matched_ids:
        return []

    matched_texts = results["documents"][0]
    matched_metadata = results["metadatas"][0]
    matched_distances = results["distances"][0]

    return [
        {
            "text": text,
            "metadata": metadata,
            "distance": distance
        }
        for text, metadata, distance in zip(
            matched_texts,
            matched_metadata,
            matched_distances
        )
    ]


def delete_document_chunks(document_id: str):

    collection.delete(
        where={"document_id": document_id}
    )


def get_document_metadata(document_id: str):

    results = collection.get(
        where={"document_id": document_id},
        include=["metadatas"]
    )

    if not results["ids"]:
        return None

    metadata = results["metadatas"][0]

    return {
        "id": document_id,
        "original_filename": metadata["filename"],
        "total_pages": metadata["total_pages"],
        "total_chunks": len(results["ids"])
    }


def list_indexed_documents():

    results = collection.get(
        include=["metadatas"]
    )

    documents = {}

    for metadata in results["metadatas"]:

        document_id = metadata["document_id"]

        if document_id not in documents:
            documents[document_id] = {
                "id": document_id,
                "original_filename": metadata["filename"],
                "total_pages": metadata["total_pages"],
                "total_chunks": 0
            }

        documents[document_id]["total_chunks"] += 1

    return list(documents.values())