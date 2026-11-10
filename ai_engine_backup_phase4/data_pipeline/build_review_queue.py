import csv
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRAFTS = ROOT / "dataset/captions/caption_drafts.jsonl"
OUTPUT = ROOT / "dataset/metadata/dataset.csv"

def split_for_hash(value):
    number = int(value[:8], 16) % 100
    if number < 90:
        return "train"
    if number < 95:
        return "validation"
    return "test"

def main():
    if not DRAFTS.exists():
        print("Caption drafts not found.")
        return

    rows = []
    seen_hashes = set()

    with DRAFTS.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            relative_path = item["image_path"]
            full_path = ROOT / relative_path

            if not full_path.is_file():
                print("Missing image:", relative_path)
                continue

            digest = hashlib.sha256(full_path.read_bytes()).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)

            rows.append({
                "image_id": full_path.stem,
                "image_path": relative_path,
                "caption_en": item["caption_en_draft"],
                "caption_hi": "",
                "source": "UNVERIFIED",
                "source_url": "",
                "license": "UNVERIFIED",
                "license_url": "",
                "creator": "UNVERIFIED",
                "attribution": "",
                "commercial_use_allowed": "no",
                "legal_review": "no",
                "category": full_path.parent.name,
                "split": split_for_hash(digest),
                "sha256": digest,
                "review_status": "development_only"
            })

    fields = [
        "image_id", "image_path", "caption_en", "caption_hi",
        "source", "source_url", "license", "license_url",
        "creator", "attribution", "commercial_use_allowed",
        "legal_review", "category", "split", "sha256",
        "review_status"
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("Development review queue created:", OUTPUT)
    print("Development images:", len(rows))
    print("Legal approval: NONE")

if __name__ == "__main__":
    main()
