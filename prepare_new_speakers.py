import os
import json
import soundfile as sf
import numpy as np
import parselmouth
import textgrid

speakers = ["ANKR29", "KAPO24", "TASO15", "VIST09", "YULDE10"]

output_dir = "/Users/karina/Desktop/university/whisper_dataset_new"
audio_dir = os.path.join(output_dir, "audio")
os.makedirs(audio_dir, exist_ok=True)

metadata = []

for speaker in speakers:
    print(f"Processing {speaker}...")
    wav_path = f"/Users/karina/Desktop/university/corpus/to_align/{speaker}/{speaker}.wav"
    tg_path = f"/Users/karina/Desktop/university/corpus/output_new/{speaker}/{speaker}.TextGrid"

    try:
        sound = parselmouth.Sound(wav_path)
        tg = textgrid.TextGrid.fromFile(tg_path)
    except Exception as e:
        print(f"  Error: {e}")
        continue

    words_tier = tg[0]
    sr = int(sound.sampling_frequency)

    chunks = []
    current_chunk = []
    chunk_start = None

    for interval in words_tier:
        word = interval.mark.strip()
        if word == "" or word == "<unk>" or word == "spn":
            continue
        if chunk_start is None:
            chunk_start = interval.minTime
        current_chunk.append((word, interval.minTime, interval.maxTime))
        if interval.maxTime - chunk_start > 10:
            chunks.append((chunk_start, interval.maxTime, current_chunk))
            current_chunk = []
            chunk_start = None

    if current_chunk:
        chunks.append((chunk_start, current_chunk[-1][2], current_chunk))

    for i, (start, end, words) in enumerate(chunks):
        segment = sound.extract_part(from_time=start, to_time=end, preserve_times=False)
        samples = segment.values[0].astype(np.float32)
        filename = f"{speaker}_chunk{i:03d}.wav"
        sf.write(os.path.join(audio_dir, filename), samples, sr)
        text = " ".join([w[0] for w in words])
        metadata.append({
            "file": filename,
            "speaker": speaker,
            "text": text,
            "start": round(start, 4),
            "end": round(end, 4)
        })

    print(f"  {len(chunks)} chunks")

with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"\nTotal chunks: {len(metadata)}")