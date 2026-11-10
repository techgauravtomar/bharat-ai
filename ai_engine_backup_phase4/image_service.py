import torch
import threading
import uuid

from pathlib import Path
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "storage" / "generated"

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

_pipeline = None
_generation_lock = threading.Lock()


def load_pipeline():
    global _pipeline

    if _pipeline is None:
        if not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS is not available.")

        print("Loading Bharat AI model...")

        _pipeline = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            use_safetensors=True
        )

        _pipeline = _pipeline.to("mps")
        _pipeline.enable_attention_slicing()

        print("Bharat AI model loaded.")

    return _pipeline


def generate_image(prompt, negative_prompt=None):
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with _generation_lock:
        pipe = load_pipeline()

        with torch.inference_mode():
            result = pipe(
                prompt=prompt.strip(),
                negative_prompt=negative_prompt or None,
                height=512,
                width=512,
                num_inference_steps=20,
                guidance_scale=7.0
            )

        filename = f"bharat_ai_{uuid.uuid4().hex}.png"
        output_path = OUTPUT_DIR / filename

        result.images[0].save(output_path)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return filename
