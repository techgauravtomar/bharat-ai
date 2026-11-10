import torch
from pathlib import Path
from datetime import datetime
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "storage" / "generated"

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.backends.mps.is_available():
        print("Apple MPS is not available.")
        return

    print("\n================================")
    print("       BHARAT AI GENERATOR")
    print("================================\n")

    prompt = input("Enter your image prompt: ").strip()

    if not prompt:
        print("Prompt cannot be empty.")
        return

    negative_prompt = input(
        "Negative prompt (optional): "
    ).strip()

    print("\nLoading AI model...")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    pipe = pipe.to("mps")
    pipe.enable_attention_slicing()

    print("Generating your image...")

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            height=512,
            width=512,
            num_inference_steps=20,
            guidance_scale=7.0
        )

    filename = datetime.now().strftime(
        "bharat_ai_%Y%m%d_%H%M%S.png"
    )

    output_path = OUTPUT_DIR / filename
    result.images[0].save(output_path)

    print("\nImage generated successfully!")
    print("Saved at:", output_path)

    torch.mps.empty_cache()

if __name__ == "__main__":
    main()
