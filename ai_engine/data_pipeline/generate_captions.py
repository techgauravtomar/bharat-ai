import json
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "dataset/raw/india"
OUTPUT = ROOT / "dataset/captions/caption_drafts.jsonl"

MODEL_NAME = "Salesforce/blip-image-captioning-base"

def main():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Loading caption model on:", device)
    print("First run may download model weights.")

    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    image_files = [
        p for p in IMAGE_DIR.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]

    print("Images found:", len(image_files))

    with OUTPUT.open("w", encoding="utf-8") as output:
        for index, path in enumerate(image_files, start=1):
            try:
                image = Image.open(path).convert("RGB")
                inputs = processor(
                    images=image,
                    return_tensors="pt"
                ).to(device)

                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=60
                    )

                caption = processor.decode(
                    generated[0],
                    skip_special_tokens=True
                )

                record = {
                    "image_path": str(path.relative_to(ROOT)),
                    "caption_en_draft": caption,
                    "human_review_required": True,
                    "review_status": "pending"
                }

                output.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                print(f"{index}/{len(image_files)}:", caption)

            except Exception as error:
                print("Could not caption:", path.name, error)

    print("\nCaption drafts saved at:", OUTPUT)

if __name__ == "__main__":
    main()
