from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from ai_engine.image_service import generate_image

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "storage" / "generated"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Bharat AI API",
    description="Indian Text-to-Image AI Platform",
    version="0.3.0"
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

app.mount(
    "/generated",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="generated"
)


class ImageRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    negative_prompt: str | None = Field(
        default=None,
        max_length=1000
    )


@app.get("/")
def home():
    return {
        "name": "Bharat AI",
        "message": "Welcome to Bharat AI API",
        "version": "0.3.0",
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
        "ai_engine": "configured",
        "model": "Stable Diffusion 1.5",
        "device": "Apple MPS",
        "message": "Image generation API is ready."
    }


@app.post("/api/v1/images/generate")
def create_image(request: ImageRequest):
    try:
        filename = generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt
        )

        return {
            "success": True,
            "message": "Image generated successfully",
            "filename": filename,
            "image_url": f"/generated/{filename}"
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        print("Image generation error:", repr(error))
        raise HTTPException(
            status_code=500,
            detail="Image generation failed. Check backend logs."
        )
