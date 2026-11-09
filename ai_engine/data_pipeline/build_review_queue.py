import csv
import json
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = ROOT / "dataset/metadata/download_manifest.csv"
LOG = ROOT / "dataset/metadata/download_log.csv"
DRAFTS = ROOT / "dataset/captions/caption_drafts.jsonl"
OUTPUT = ROOT / "dataset/metadata/dataset.csv"

def safe_name(value):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)[:80]

def split_for_hash(value):
    number = int(value[:8], 16) % 100
    if number < 90:
        return "train"
    if number < 95:
        return "validation"
    return "test"

def main():
    if not all(p.exists() for p in [MANIFEST, LOG, DRAFTS]):
        print("Run download and captioning steps first.")
        return

    with LOG.open(encoding="utf-8", newline="") as f:
        logs = {r["image_id"]: r for r in csv.DictReader(f)}

    drafts = {}
    with DRAFTS.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            drafts[item["image_path"]] = item["caption_en_draft"]

    output_rows = []
    seen_hashes = set()

    with MANIFEST.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            image_id = safe_name(row["image_id"])
            category = safe_name(row["category"])

            log = logs.get(image_id, {})
            if log.get("status") != "DOWNLOADED":
                continue

            relative_path = (
                Path("dataset/raw/india") /
                category / f"{image_id}.jpg"
            )
            full_path = ROOT / relative_path

            if not full_path.is_file():
                continue

            digest = hashlib.sha256(full_path.read_bytes()).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)

            key = relative_path.as_posix()
            caption = drafts.get(key, "")

            output_rows.append({
                "image_id": image_id,
                "image_path": key,
                "caption_en": caption,
                "caption_hi": "",
                "source": row["source_name"],
                "source_url": row["source_page_url"],
                "license": row["license"],
                "license_url": row["license_url"],
                "creator": row["creator"],
                "attribution": row["attribution"],
                "commercial_use_allowed": row["commercial_use_allowed"],
                "legal_review": row["legal_review"],
                "category": category,
                "split": split_for_hash(digest),
                "sha256": digest,
                "review_status": "pending",
            })

    fields = [
        "image_id", "image_path", "caption_en", "caption_hi",
        "source", "source_url", "license", "license_url",
        "creator", "attribution", "commercial_use_allowed",
        "legal_review", "category", "split", "sha256",
        "review_status"
    ]

    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    print("Review queue created:", OUTPUT)
    print("Records:", len(output_rows))
    print("Review captions and rights before approval.")

if __name__ == "__main__":
    main()
