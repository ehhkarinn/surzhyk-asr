import os
import json
import shutil

original_dir = "/Users/karina/Desktop/university/whisper_dataset"
new_dir = "/Users/karina/Desktop/university/whisper_dataset_new"
combined_dir = "/Users/karina/Desktop/university/whisper_dataset_combined"
audio_dir = os.path.join(combined_dir, "audio")
os.makedirs(audio_dir, exist_ok=True)

with open(os.path.join(original_dir, "metadata.json"), "r", encoding="utf-8") as f:
    original_metadata = json.load(f)

new_speakers = ["ANKR29", "KAPO24", "TASO15", "VIST09"]
with open(os.path.join(new_dir, "metadata.json"), "r", encoding="utf-8") as f:
    new_metadata = [item for item in json.load(f) if item["speaker"] in new_speakers]

for item in original_metadata:
    src = os.path.join(original_dir, "audio", item["file"])
    shutil.copy(src, os.path.join(audio_dir, item["file"]))

for item in new_metadata:
    src = os.path.join(new_dir, "audio", item["file"])
    shutil.copy(src, os.path.join(audio_dir, item["file"]))

combined = original_metadata + new_metadata

with open(os.path.join(combined_dir, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)

print(f"Original chunks: {len(original_metadata)}")
print(f"New chunks added: {len(new_metadata)}")
print(f"Total combined: {len(combined)}")
