import os
import json
import soundfile as sf
import numpy as np
import parselmouth
import textgrid

speakers = ["ANKU", "ANMO17"]

output_dir = "/Users/karina/Desktop/university/whisper_dataset_unseen"
audio_dir = os.path.join(output_dir, "audio")
os.makedirs(audio_dir, exist_ok=True)

metadata = []

for speaker in speakers:
    print(f"Processing {speaker}...")
    wav_path = f"/Users/karina/Desktop/university/corpus/{speaker}/{speaker}.wav"
    tg_path = f"/Users/karina/Desktop/university/corpus/output/{speaker}/{speaker}.TextGrid"

    sound = parselmouth.Sound(wav_path)
    tg = textgrid.TextGrid.fromFile(tg_path)
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

    print(f"  Found {len(chunks)} chunks")

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

with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"Total chunks: {len(metadata)}")