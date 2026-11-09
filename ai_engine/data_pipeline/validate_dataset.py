import csv
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_FILE = PROJECT_ROOT / "dataset/metadata/dataset.csv"

REQUIRED_FIELDS = [
    "image_path",
    "caption",
    "source",
    "source_url",
    "license",
    "commercial_use_allowed",
    "category",
    "split",
]

VALID_SPLITS = {"train", "validation", "test"}

def validate():
    if not CSV_FILE.exists():
        print("ERROR: Dataset CSV not found.")
        return

    errors = []
    valid_count = 0

    with CSV_FILE.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        missing = set(REQUIRED_FIELDS) - set(reader.fieldnames or [])
        if missing:
            print("Missing columns:", ", ".join(sorted(missing)))
            return

        for row_number, row in enumerate(reader, start=2):
            issues = []

            for field in REQUIRED_FIELDS:
                if not (row.get(field) or "").strip():
                    issues.append(f"Missing {field}")

            image_value = (row.get("image_path") or "").strip()
            if image_value:
                image_path = PROJECT_ROOT / image_value

                if not image_path.is_file():
                    issues.append("Image file not found")
                else:
                    try:
                        with Image.open(image_path) as image:
                            image.verify()
                    except Exception:
                        issues.append("Invalid or unreadable image")

            permission = (
                row.get("commercial_use_allowed") or ""
            ).strip().lower()

            if permission != "yes":
                issues.append(
                    "Commercial-use permission not confirmed"
                )

            split = (row.get("split") or "").strip().lower()
            if split not in VALID_SPLITS:
                issues.append("Invalid dataset split")

            if issues:
                errors.append((row_number, issues))
            else:
                valid_count += 1

    print(f"\nValid records: {valid_count}")
    print(f"Records with errors: {len(errors)}")

    for row_number, issues in errors:
        print(f"Row {row_number}: {'; '.join(issues)}")

    if not errors:
        print("Dataset validation successful.")

if __name__ == "__main__":
    validate()
