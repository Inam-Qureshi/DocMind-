
import pymupdf
from pathlib import Path


def extract_text_from_pdf(file_path: str) -> list[dict]:

    extracted_pages = []

    pdf_path = Path(file_path)

    with pymupdf.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            page_text = page.get_text("text").strip()

            if page_text:
                extracted_pages.append({
                    "page_number": page_number,
                    "text": page_text
                })

    return extracted_pages
