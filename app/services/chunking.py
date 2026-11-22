
def split_text_into_chunks(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[dict]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "page_number": page_number,
                    "text": chunk_text
                })
                chunk_id += 1

            start += chunk_size - chunk_overlap

    return chunks
