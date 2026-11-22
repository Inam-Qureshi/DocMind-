
from fastapi import FastAPI
from app.routes.documents import router as documents_router

app = FastAPI(
    title="DocuMind",
    description="A PDF-based RAG chatbot",
    version="1.0.0"
)

app.include_router(documents_router)


@app.get("/")
def home():
    return {"message": "Welcome to DocuMind!"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
