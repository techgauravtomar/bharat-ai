import csv
import hashlib
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CSV_FILE = ROOT / "dataset/metadata/dataset.csv"

REQUIRED = [
    "image_id", "image_path", "caption_en",
    "source", "source_url", "license", "license_url",
    "commercial_use_allowed", "legal_review",
    "category", "split", "sha256", "review_status"
]

ALLOWED_LICENSES = {
    "CC0-1.0",
    "Public Domain",
    "GODL-India",
    "Direct Permission",
}

def calculate_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    if not CSV_FILE.exists():
        print("Dataset CSV not found.")
        return

    errors = []
    seen_hashes = set()
    valid = 0

    with CSV_FILE.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        missing = set(REQUIRED) - set(reader.fieldnames or [])
        if missing:
            print("Missing columns:", sorted(missing))
            return

        for line, row in enumerate(reader, start=2):
            issues = []

            for field in REQUIRED:
                if not (row.get(field) or "").strip():
                    issues.append(f"Missing {field}")

            path = ROOT / (row.get("image_path") or "")

            if not path.is_file():
                issues.append("Image file missing")
            else:
                try:
                    with Image.open(path) as image:
                        image.verify()

                    actual_hash = calculate_hash(path)
                    recorded_hash = row["sha256"].strip()

                    if actual_hash != recorded_hash:
                        issues.append("SHA256 mismatch")

                    if actual_hash in seen_hashes:
                        issues.append("Duplicate image")

                    seen_hashes.add(actual_hash)

                except Exception:
                    issues.append("Invalid image")

            if row.get("license", "").strip() not in ALLOWED_LICENSES:
                issues.append("License not allowlisted")

            if row.get("commercial_use_allowed", "").lower() != "yes":
                issues.append("Commercial use not approved")

            if row.get("legal_review", "").lower() != "yes":
                issues.append("Legal review pending")

            if row.get("review_status", "").lower() != "approved":
                issues.append("Caption review pending")

            if row.get("split", "").lower() not in {
                "train", "validation", "test"
            }:
                issues.append("Invalid split")

            if issues:
                errors.append((line, issues))
            else:
                valid += 1

    print("\nValid records:", valid)
    print("Records requiring correction:", len(errors))

    for line, issues in errors:
        print(f"Row {line}: {'; '.join(issues)}")

    if not errors:
        print("Dataset validation successful.")

if __name__ == "__main__":
    main()
