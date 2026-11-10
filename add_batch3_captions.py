
import json
import shutil
from pathlib import Path

queue = Path("dataset/captions/temple_caption_queue.jsonl")
backup = Path("dataset/captions/temple_caption_queue_before_batch3.jsonl")

if not queue.exists():
    raise SystemExit("ERROR: Caption queue not found.")

rows = [
    json.loads(line)
    for line in queue.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

if len(rows) != 33:
    raise SystemExit(f"Expected 33 records, found {len(rows)}")

# Captions correspond to images 21–30 and 32–33.
captions = {
    20: (
        "An Indian temple exterior with carved architectural surfaces and traditional decorative details.",
        "भारतीय मंदिर का बाहरी दृश्य, जिसमें नक्काशीदार वास्तुशिल्पीय सतहें और पारंपरिक सजावटी विवरण दिखाई देते हैं।"
    ),
    21: (
        "A traditional Indian temple structure featuring stonework, sculptural details, and an ornamented exterior.",
        "पारंपरिक भारतीय मंदिर की संरचना, जिसमें पत्थर का काम, मूर्तिकारी विवरण और अलंकृत बाहरी भाग दिखाई देता है।"
    ),
    22: (
        "An Indian temple with a detailed exterior, carved architectural elements, and a traditional stone structure.",
        "भारतीय मंदिर, जिसमें बारीक बाहरी बनावट, नक्काशीदार वास्तुशिल्पीय तत्व और पारंपरिक पत्थर की संरचना दिखाई देती है।"
    ),
    23: (
        "A historic Indian temple displaying sculpted stone details and decorative architectural features.",
        "ऐतिहासिक भारतीय मंदिर, जिसमें पत्थर की मूर्तिकारी और सजावटी वास्तुशिल्पीय विशेषताएँ दिखाई देती हैं।"
    ),
    24: (
        "A traditional temple exterior with carved surfaces, ornamental details, and a stone architectural form.",
        "पारंपरिक मंदिर का बाहरी भाग, जिसमें नक्काशीदार सतहें, सजावटी विवरण और पत्थर की वास्तुशिल्पीय संरचना दिखाई देती है।"
    ),
    25: (
        "An Indian temple facade with sculptural ornamentation and detailed traditional architectural features.",
        "भारतीय मंदिर का अग्रभाग, जिसमें मूर्तिकारी सजावट और पारंपरिक वास्तुकला के बारीक विवरण दिखाई देते हैं।"
    ),
    26: (
        "A stone temple featuring a decorated tower and detailed architectural elements across its exterior.",
        "पत्थर का मंदिर, जिसमें सजावटी शिखर और बाहरी भाग पर विस्तृत वास्तुशिल्पीय तत्व दिखाई देते हैं।"
    ),
    27: (
        "A traditional Indian temple with carved stonework and an intricately decorated exterior.",
        "पारंपरिक भारतीय मंदिर, जिसमें नक्काशीदार पत्थर का काम और बारीकी से सजाया गया बाहरी भाग दिखाई देता है।"
    ),
    28: (
        "An Indian temple structure showing traditional stone architecture and ornamental sculptural details.",
        "भारतीय मंदिर की संरचना, जिसमें पारंपरिक पत्थर की वास्तुकला और सजावटी मूर्तिकारी विवरण दिखाई देते हैं।"
    ),
    29: (
        "A view of an Indian temple with a traditional architectural form and visible decorative details.",
        "भारतीय मंदिर का दृश्य, जिसमें पारंपरिक वास्तुशिल्पीय स्वरूप और दिखाई देने वाले सजावटी विवरण मौजूद हैं।"
    ),
    31: (
        "The Siddheshwar Temple in Bhubaneswar, Odisha, identified in the source filename as having a Pancharatna-style tower.",
        "ओडिशा के भुवनेश्वर स्थित सिद्धेश्वर मंदिर, जिसके स्रोत filename में शिखर को पंचरत्न शैली का बताया गया है।"
    ),
    32: (
        "A historic Indian temple in a traditional architectural style, featuring a decorated tower and detailed stone elements.",
        "पारंपरिक वास्तुशिल्पीय शैली में बना ऐतिहासिक भारतीय मंदिर, जिसमें सजावटी शिखर और पत्थर के बारीक विवरण दिखाई देते हैं।"
    )
}

# Verify target records are empty and images exist.
for index in list(range(20, 30)) + [31, 32]:
    if rows[index].get("caption_en") or rows[index].get("caption_hi"):
        raise SystemExit(f"Record {index + 1} already has captions.")

    image_path = Path(rows[index]["image_path"])
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

# Image 31 in the numbered list is a diagram, not a temple photograph.
diagram_index = 30
diagram_path = Path(rows[diagram_index]["image_path"])

if not diagram_path.is_file():
    raise SystemExit(f"Diagram not found: {diagram_path}")

shutil.copy2(queue, backup)

for index, (en, hi) in captions.items():
    rows[index]["caption_en"] = en
    rows[index]["caption_hi"] = hi
    rows[index]["category"] = "indian_temple_architecture"
    rows[index]["human_review_required"] = True
    rows[index]["review_status"] = "pending"

# Preserve the diagram as a reference, not a training image.
rows[diagram_index]["category"] = "reference_only"
rows[diagram_index]["caption_en"] = (
    "An educational diagram comparing architectural features "
    "of North Indian and South Indian temples."
)
rows[diagram_index]["caption_hi"] = (
    "उत्तर भारतीय और दक्षिण भारतीय मंदिरों की वास्तुशिल्पीय "
    "विशेषताओं की तुलना करने वाला शैक्षिक चित्र।"
)
rows[diagram_index]["human_review_required"] = True
rows[diagram_index]["review_status"] = "excluded_from_training"

with queue.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Batch 3 temple captions added:", len(captions))
print("Reference-only diagram:", diagram_path.name)
print("Total queue records:", len(rows))
print("Backup:", backup)
