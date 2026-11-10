import csv
import hashlib
import sys
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
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if not CSV_FILE.exists():
        print("Dataset CSV not found.")
        sys.exit(1)

    errors = []
    seen_hashes = set()
    development_valid = 0
    legal_valid = 0
    total = 0

    with CSV_FILE.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        missing = set(REQUIRED) - set(reader.fieldnames or [])
        if missing:
            print("Missing columns:", sorted(missing))
            sys.exit(1)

        for line, row in enumerate(reader, start=2):
            total += 1
            issues = []

            for field in REQUIRED:
                if field in {"source_url", "license_url"}:
                    continue
                if not (row.get(field) or "").strip():
                    issues.append(f"Missing {field}")

            image_path = ROOT / (row.get("image_path") or "")

            if not image_path.is_file():
                issues.append("Image file missing")
            else:
                try:
                    with Image.open(image_path) as image:
                        image.verify()

                    actual_hash = calculate_hash(image_path)
                    recorded_hash = (row.get("sha256") or "").strip()

                    if actual_hash != recorded_hash:
                        issues.append("SHA256 mismatch")

                    if actual_hash in seen_hashes:
                        issues.append("Duplicate image")

                    seen_hashes.add(actual_hash)

                except Exception:
                    issues.append("Invalid image")

            if (row.get("split") or "").lower() not in {
                "train", "validation", "test"
            }:
                issues.append("Invalid split")

            status = (row.get("review_status") or "").strip().lower()

            if status == "development_only":
                if row.get("commercial_use_allowed", "").lower() != "no":
                    issues.append("Development image must not be commercially approved")

                if row.get("legal_review", "").lower() != "no":
                    issues.append("Development image must not have legal approval")

                if row.get("license", "").strip() != "UNVERIFIED":
                    issues.append("Development license must remain UNVERIFIED")

                if not issues:
                    development_valid += 1

            elif status == "approved":
                if row.get("license", "").strip() not in ALLOWED_LICENSES:
                    issues.append("License not allowlisted")

                if row.get("commercial_use_allowed", "").lower() != "yes":
                    issues.append("Commercial use not approved")

                if row.get("legal_review", "").lower() != "yes":
                    issues.append("Legal review pending")

                if not row.get("source_url", "").strip():
                    issues.append("Source URL missing")

                if not row.get("license_url", "").strip():
                    issues.append("License URL missing")

                if not issues:
                    legal_valid += 1

            else:
                issues.append("Unknown or pending review status")

            if issues:
                errors.append((line, issues))

    print("\nTotal records:", total)
    print("Valid development records:", development_valid)
    print("Valid legal records:", legal_valid)
    print("Records requiring correction:", len(errors))

    for line, issues in errors:
        print(f"Row {line}: {'; '.join(issues)}")

    if total == 0:
        print("ERROR: Dataset is empty.")
        sys.exit(1)

    if errors:
        print("Validation failed.")
        sys.exit(1)

    print("Dataset validation successful.")


if __name__ == "__main__":
    main()
