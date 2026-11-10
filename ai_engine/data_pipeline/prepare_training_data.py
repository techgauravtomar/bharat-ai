import json
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CAPTIONS = ROOT / "dataset/captions/captions.jsonl"
OUTPUT = ROOT / "dataset/metadata/training_manifest.jsonl"

def main():
    if not CAPTIONS.exists():
        print("Captions file missing.")
        sys.exit(1)

    records = []
    errors = []

    for line_no, line in enumerate(
        CAPTIONS.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            item = json.loads(line)
            image_path = ROOT / item["image_path"]
            caption = item.get("caption_en", "").strip()

            if not image_path.is_file():
                raise ValueError("Image file missing")

            if not caption:
                raise ValueError("English caption missing")

            with Image.open(image_path) as image:
                image.verify()

            records.append({
                "image_path": item["image_path"],
                "caption": caption,
                "category": item.get("category", "indian_architecture"),
                "review_status": item.get("review_status", "pending"),
                "human_review_required": item.get(
                    "human_review_required", True
                )
            })

        except Exception as error:
            errors.append(f"Line {line_no}: {error}")

    print("Records found:", len(records))
    print("Errors:", len(errors))

    for error in errors:
        print(error)

    if errors or not records:
        print("Preparation failed.")
        sys.exit(1)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8"
    )

    print("Manifest created:", OUTPUT)
    print("NOTE: Pending captions and unverified image rights")
    print("must be reviewed before actual model training.")

if __name__ == "__main__":
    main()
