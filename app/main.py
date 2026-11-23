
from fastapi import FastAPI

from app.routes.documents import router as documents_router
from app.routes.chat import router as chat_router

app = FastAPI(
    title="DocMind",
    description="A PDF-based RAG chatbot",
    version="1.0.0"
)

# Register API routers
app.include_router(documents_router)
app.include_router(chat_router)

@app.get("/")
def home():

    return {
        "message": "Welcome to DocuMind!"
    }

@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }