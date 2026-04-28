import pandas as pd
import json
import os

features_df = pd.read_csv("/Users/karina/Desktop/university/features_all_speakers.csv")
print(f"Word-level features: {len(features_df)} words")
print(f"Columns: {list(features_df.columns)}")
print()

with open("/Users/karina/Desktop/university/whisper_dataset_9speakers/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)
print(f"Total chunks: {len(metadata)}")

chunk_features = []

for chunk in metadata:
    speaker = chunk["speaker"]
    chunk_start = chunk["start"]
    chunk_end = chunk["end"]

    mask = (
        (features_df["speaker"] == speaker) &
        (features_df["start"] >= chunk_start) &
        (features_df["end"] <= chunk_end)
    )
    chunk_words = features_df[mask]

    if len(chunk_words) == 0:
        print(f"Warning: no words found for {speaker} chunk {chunk_start}-{chunk_end}")
        continue

    mfcc_cols = [f"mfcc_{i}" for i in range(1, 14)]

    row = {
        "file": chunk["file"],
        "speaker": speaker,
        "text": chunk["text"],
        "start": chunk_start,
        "end": chunk_end,
        "n_words": len(chunk_words),
        "mean_pitch": chunk_words["pitch_mean"].mean(),
        "std_pitch": chunk_words["pitch_mean"].std(),
        "mean_f1": chunk_words["f1_mean"].mean(),
        "mean_f2": chunk_words["f2_mean"].mean(),
    }

    for col in mfcc_cols:
        if col in chunk_words.columns:
            row[f"mean_{col}"] = chunk_words[col].mean()
            row[f"std_{col}"] = chunk_words[col].std()

    chunk_features.append(row)

chunk_df = pd.DataFrame(chunk_features)
output_path = "/Users/karina/Desktop/university/chunk_features.csv"
chunk_df.to_csv(output_path, index=False)

print(f"\nDone! {len(chunk_df)} chunks with aggregated features")
print(f"Saved to {output_path}")
print()
print("Preview:")
print(chunk_df[["speaker", "file", "n_words", "mean_pitch", "mean_f1", "mean_f2", "mean_mfcc_1"]].head(10))
