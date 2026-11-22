
from pathlib import Path
from uuid import uuid4
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.pdf_service import extract_text_from_pdf
from app.services.chunking_service import split_text_into_chunks


router = APIRouter(
    prefix="/documents",
    tags=["Documents"]
)

UPLOAD_DIR = Path(__file__).resolve().parents[2] / "storage" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Temporary in-memory document storage
documents = []


@router.post("/upload")
def upload_pdf(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_signature = file.file.read(4)
    file.file.seek(0)

    if file_signature != b"%PDF":
        raise HTTPException(status_code=400, detail="Invalid PDF file.")

    document_id = str(uuid4())
    stored_filename = f"{document_id}.pdf"
    file_path = UPLOAD_DIR / stored_filename

    try:
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

        # Store document information
        document = {
            "id": document_id,
            "original_filename": file.filename,
            "stored_filename": stored_filename,
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "extracted_pages": pages,
            "chunks": chunks
        }

        documents.append(document)

        return {
            "message": "PDF uploaded, text extracted, and chunked successfully.",
            "document": document
        }

    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise

    except Exception as error:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(error)}"
        )

    finally:
        file.file.close()


@router.get("/")
def get_documents():
    return {
        "total_documents": len(documents),
        "documents": documents
    }


@router.get("/{document_id}")
def get_document(document_id: str):

    for document in documents:
        if document["id"] == document_id:
            return document

    raise HTTPException(status_code=404, detail="Document not found.")


@router.delete("/{document_id}")
def delete_document(document_id: str):

    for document in documents:
        if document["id"] == document_id:
            file_path = UPLOAD_DIR / document["stored_filename"]

            if file_path.exists():
                file_path.unlink()

            documents.remove(document)

            return {"message": "Document deleted successfully"}

    raise HTTPException(status_code=404, detail="Document not found.")
