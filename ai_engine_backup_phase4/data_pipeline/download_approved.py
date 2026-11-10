import csv
import hashlib
import re
import urllib.request
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "dataset/metadata/download_manifest.csv"
OUTPUT = ROOT / "dataset/raw/india"
LOG = ROOT / "dataset/metadata/download_log.csv"

ALLOWED_LICENSES = {
    "CC0-1.0",
    "Public Domain",
    "GODL-India",
    "Direct Permission",
}

MAX_BYTES = 20 * 1024 * 1024

def safe_name(value):
    value = re.sub(r"[^a-zA-Z0-9_-]", "_", value)
    return value[:80]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    if not MANIFEST.exists():
        print("Manifest not found:", MANIFEST)
        return

    logs = []

    with MANIFEST.open(encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)

        for row in rows:
            image_id = safe_name(row.get("image_id", ""))
            category = safe_name(row.get("category", ""))
            url = row.get("download_url", "").strip()
            source = row.get("source_page_url", "").strip()
            license_name = row.get("license", "").strip()
            license_url = row.get("license_url", "").strip()

            checks = [
                row.get("approved", "").lower() == "yes",
                row.get("commercial_use_allowed", "").lower() == "yes",
                row.get("legal_review", "").lower() == "yes",
                license_name in ALLOWED_LICENSES,
                bool(image_id and category and url and source and license_url),
                url.startswith("https://"),
            ]

            if not all(checks):
                logs.append([image_id, "SKIPPED", "Approval or metadata incomplete"])
                continue

            folder = OUTPUT / category
            folder.mkdir(parents=True, exist_ok=True)

            temp_path = folder / f"{image_id}.download"
            final_path = folder / f"{image_id}.jpg"

            try:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "BharatAI-Dataset/0.1"}
                )

                with urllib.request.urlopen(request, timeout=30) as response:
                    content_type = response.headers.get("Content-Type", "")
                    if not content_type.startswith("image/"):
                        raise ValueError("URL did not return an image")

                    data = response.read(MAX_BYTES + 1)

                if len(data) > MAX_BYTES:
                    raise ValueError("Image exceeds 20 MB limit")

                temp_path.write_bytes(data)

                with Image.open(temp_path) as image:
                    image.verify()

                temp_path.replace(final_path)
                digest = sha256(final_path)

                logs.append([image_id, "DOWNLOADED", digest])
                print("Downloaded:", final_path.relative_to(ROOT))

            except Exception as error:
                temp_path.unlink(missing_ok=True)
                logs.append([image_id, "FAILED", str(error)])
                print("Failed:", image_id, error)

    with LOG.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "status", "details"])
        writer.writerows(logs)

    print("\nDownload process finished.")
    print("Review:", LOG)

if __name__ == "__main__":
    main()
