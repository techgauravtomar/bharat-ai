from fastapi import FastAPI

app = FastAPI(
    title="Bharat AI API",
    description="Indian Text-to-Image AI Platform",
    version="0.1.0"
)

@app.get("/")
def home():
    return {
        "name": "Bharat AI",
        "status": "running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }