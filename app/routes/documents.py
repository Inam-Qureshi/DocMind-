from pathlib import Path
from uuid import uuid4
from contextlib import suppress
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.pdf_service import extract_text_from_pdf
from app.services.chunking import split_text_into_chunks
from app.services.embeddings import generate_embeddings

from app.services.vector_store import (
    add_chunks,
    delete_document_chunks,
    get_document_metadata,
    list_indexed_documents
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"]
)


UPLOAD_DIR = Path(__file__).resolve().parents[2] / "storage" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
def upload_pdf(file: UploadFile = File(...)):

    document_id = str(uuid4())
    stored_filename = f"{document_id}.pdf"
    file_path = UPLOAD_DIR / stored_filename

    try:

        # Validate filename
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed."
            )

        # Validate basic PDF signature
        file_signature = file.file.read(4)
        file.file.seek(0)

        if file_signature != b"%PDF":
            raise HTTPException(
                status_code=400,
                detail="Invalid PDF file."
            )

        # Save uploaded PDF
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text from PDF
        pages = extract_text_from_pdf(str(file_path))

        if not pages:
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in this PDF."
            )

        # Split extracted text into chunks
        chunks = split_text_into_chunks(pages)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text chunks could be generated."
            )

        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]

        embeddings = generate_embeddings(texts)

        # Store chunks, embeddings and metadata in ChromaDB
        add_chunks(
            document_id=document_id,
            filename=file.filename,
            total_pages=len(pages),
            chunks=chunks,
            embeddings=embeddings
        )

        document = get_document_metadata(document_id)

        return {
            "message": "PDF uploaded and indexed successfully.",
            "document": document
        }

    except HTTPException:

        file_path.unlink(missing_ok=True)

        with suppress(Exception):
            delete_document_chunks(document_id)

        raise

    except Exception as error:

        file_path.unlink(missing_ok=True)

        with suppress(Exception):
            delete_document_chunks(document_id)

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(error)}"
        )

    finally:
        file.file.close()


@router.get("/")
def get_documents():

    documents = list_indexed_documents()

    return {
        "total_documents": len(documents),
        "documents": documents
    }


@router.get("/{document_id}")
def get_document(document_id: str):

    document = get_document_metadata(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    return document


@router.delete("/{document_id}")
def delete_document(document_id: str):

    document = get_document_metadata(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    file_path = UPLOAD_DIR / f"{document_id}.pdf"

    # Delete chunks and embeddings from ChromaDB
    delete_document_chunks(document_id)

    # Delete original PDF
    if file_path.exists():
        file_path.unlink()

    return {
        "message": "Document deleted successfully."
    }