

import json
import shutil
from pathlib import Path

queue = Path("dataset/captions/temple_caption_queue.jsonl")
backup = Path("dataset/captions/temple_caption_queue_before_batch2.jsonl")

if not queue.exists():
    raise SystemExit("ERROR: Caption queue file not found.")

rows = [
    json.loads(line)
    for line in queue.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

captions = [
    (
        "The Vaital Deula Temple in Bhubaneswar, Odisha, featuring a distinctive Khakhara-style roof, stone construction, and traditional Kalinga architectural details.",
        "ओडिशा के भुवनेश्वर स्थित वैताल देउल मंदिर, जिसमें विशिष्ट खाखरा शैली की छत, पत्थर की संरचना और पारंपरिक कलिंग वास्तुकला के विवरण दिखाई देते हैं।"
    ),
    (
        "An Indian temple exterior showing its stone structure, raised architectural elements, and decorative details.",
        "भारतीय मंदिर का बाहरी दृश्य, जिसमें पत्थर की संरचना, ऊँचे वास्तुशिल्पीय भाग और सजावटी विवरण दिखाई देते हैं।"
    ),
    (
        "A carved stone pillar at the Airavateshwarar Temple in Darasuram, featuring detailed sculptural work and ornamental motifs.",
        "दारासुरम के ऐरावतेश्वर मंदिर का नक्काशीदार पत्थर का स्तंभ, जिस पर विस्तृत मूर्तिकारी और सजावटी आकृतियाँ बनी हैं।"
    ),
    (
        "A view of an Indian stone temple with a richly detailed exterior and traditional architectural ornamentation.",
        "भारतीय पत्थर के मंदिर का दृश्य, जिसमें बारीक विवरणों वाला बाहरी भाग और पारंपरिक वास्तुशिल्पीय सजावट दिखाई देती है।"
    ),
    (
        "An ornate Indian temple exterior featuring carved stone surfaces, sculptural details, and decorative architectural elements.",
        "भारतीय मंदिर का अलंकृत बाहरी भाग, जिसमें नक्काशीदार पत्थर की सतह, मूर्तिकारी विवरण और सजावटी वास्तुशिल्पीय तत्व दिखाई देते हैं।"
    ),
    (
        "A traditional temple structure with detailed stonework and layered architectural features across its exterior.",
        "पारंपरिक मंदिर की संरचना, जिसके बाहरी भाग पर बारीक पत्थर का काम और परतदार वास्तुशिल्पीय बनावट दिखाई देती है।"
    ),
    (
        "An Indian temple facade displaying carved architectural details, sculpted forms, and ornamental stone decoration.",
        "भारतीय मंदिर का अग्रभाग, जिस पर नक्काशीदार वास्तुशिल्पीय विवरण, मूर्तिकारी आकृतियाँ और सजावटी पत्थर का काम दिखाई देता है।"
    ),
    (
        "A stone temple with a vertically rising tower and decorative architectural features on its exterior.",
        "पत्थर का मंदिर, जिसमें ऊपर की ओर उठता शिखर और बाहरी सतह पर सजावटी वास्तुशिल्पीय विवरण दिखाई देते हैं।"
    ),
    (
        "A traditional Indian temple exterior with intricate stone carvings and a richly ornamented architectural surface.",
        "पारंपरिक भारतीय मंदिर का बाहरी भाग, जिसमें बारीक पत्थर की नक्काशी और अलंकृत वास्तुशिल्पीय सतह दिखाई देती है।"
    ),
    (
        "A historic stone temple displaying sculpted details, carved surfaces, and traditional Indian architectural elements.",
        "ऐतिहासिक पत्थर का मंदिर, जिसमें मूर्तिकारी विवरण, नक्काशीदार सतहें और पारंपरिक भारतीय वास्तुशिल्पीय तत्व दिखाई देते हैं।"
    )
]

if len(rows) != 33:
    raise SystemExit(f"Expected 33 records, found {len(rows)}")

# Confirm that records 11–20 are still empty before writing.
for index in range(10, 20):
    if rows[index].get("caption_en") or rows[index].get("caption_hi"):
        raise SystemExit(
            f"Record {index + 1} already has a caption. No changes made."
        )

# Check image paths before changing the queue.
for index in range(10, 20):
    image_path = Path(rows[index]["image_path"])
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

# Create backup, then update only records 11–20.
shutil.copy2(queue, backup)

for index, (en, hi) in enumerate(captions, start=10):
    rows[index]["caption_en"] = en
    rows[index]["caption_hi"] = hi
    rows[index]["category"] = "indian_temple_architecture"
    rows[index]["human_review_required"] = True
    rows[index]["review_status"] = "pending"

with queue.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Batch 2 captions added:", len(captions))
print("Total records:", len(rows))
print("Pending review:", sum(r["review_status"] == "pending" for r in rows))
print("Backup:", backup)
