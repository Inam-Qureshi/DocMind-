
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.embeddings import generate_query_embedding
from app.services.vector_store import search_similar_chunks
from app.services.llm_service import generate_answer


router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    document_id: str | None = None
    top_k: int = Field(default=3, ge=1, le=10)


def retrieve_chunks(request: ChatRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # Convert question into an embedding
    query_embedding = generate_query_embedding(question)

    # Retrieve relevant chunks from ChromaDB
    chunks = search_similar_chunks(
        query_embedding=query_embedding,
        top_k=request.top_k,
        document_id=request.document_id
    )

    return chunks


@router.post("/search")
def semantic_search(request: ChatRequest):

    try:

        chunks = retrieve_chunks(request)

        results = []

        for rank, chunk in enumerate(chunks, start=1):

            metadata = chunk["metadata"]

            results.append({
                "rank": rank,
                "text": chunk["text"],
                "filename": metadata["filename"],
                "page_number": metadata["page_number"],
                "document_id": metadata["document_id"],
                "distance": chunk["distance"]
            })

        return {
            "question": request.question,
            "total_results": len(results),
            "results": results
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Semantic search failed: {str(error)}"
        )


@router.post("/ask")
def ask_question(request: ChatRequest):

    try:

        # Retrieve relevant chunks
        retrieved_chunks = retrieve_chunks(request)

        if not retrieved_chunks:
            return {
                "answer": (
                    "I couldn't find that information "
                    "in the uploaded documents."
                ),
                "sources": [],
                "retrieved_chunks": 0
            }

        # Prepare context and source references
        context_parts = []
        sources = []
        seen_sources = set()

        for chunk in retrieved_chunks:

            metadata = chunk["metadata"]

            filename = metadata["filename"]
            page_number = metadata["page_number"]

            context_parts.append(
                f"""
                Source: {filename}, page {page_number}

                Content:
                {chunk["text"]}
                """
            )

            source_key = (filename, page_number)

            if source_key not in seen_sources:

                sources.append({
                    "filename": filename,
                    "page_number": page_number,
                    "document_id": metadata["document_id"]
                })

                seen_sources.add(source_key)

        context = "\n\n".join(context_parts)

        # Generate the final answer using Gemini
        answer = generate_answer(
            question=request.question,
            context=context
        )

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(retrieved_chunks)
        }

    except RuntimeError as error:

        raise HTTPException(
            status_code=503,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Question answering failed: {str(error)}"
        )
