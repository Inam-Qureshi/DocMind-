from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

# Load the model only when embeddings are first requested.
_model = None


def get_embedding_model():

    global _model

    if _model is None:
        _model = SentenceTransformer(
            MODEL_NAME,
            device="cpu"
        )

    return _model


def generate_embeddings(texts: list[str]) -> list[list[float]]:

    if not texts:
        return []

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.tolist()


def generate_query_embedding(text: str) -> list[float]:

    embeddings = generate_embeddings([text])

    return embeddings[0]