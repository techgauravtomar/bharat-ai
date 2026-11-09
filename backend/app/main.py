from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone

app = FastAPI(
    title="Bharat AI API",
    description="Indian Text-to-Image AI Platform",
    version="0.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "name": "Bharat AI",
        "message": "Welcome to Bharat AI API",
        "version": "0.2.0",
        "status": "running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Bharat AI Backend",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/status")
def api_status():
    return {
        "platform": "Bharat AI",
        "backend": "online",
        "ai_engine": "not_configured",
        "model": None,
        "message": "AI engine integration is coming in a later phase."
    }
