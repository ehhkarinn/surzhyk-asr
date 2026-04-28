import parselmouth
import numpy as np
import pandas as pd
from parselmouth.praat import call
import textgrid

sound = parselmouth.Sound("/Users/karina/Desktop/university/corpus/ANKA19/ANKA19.wav")
tg = textgrid.TextGrid.fromFile("/Users/karina/Desktop/university/corpus/output/ANKA19/ANKA19.TextGrid")

words_tier = tg[0]
rows = []

for interval in words_tier:
    word = interval.mark.strip()
    if word == "" or word == "<unk>" or word == "spn":
        continue

    start = interval.minTime
    end = interval.maxTime
    duration = end - start

    segment = sound.extract_part(from_time=start, to_time=end, preserve_times=False)

    # MFCCs using librosa
    import librosa
    samples = segment.values[0]
    sr = int(segment.sampling_frequency)
    n_fft = min(2048, len(samples))
    mfccs = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=13, n_fft=n_fft)
    mfcc_means = np.mean(mfccs, axis=1)

    # Pitch
    pitch_obj = call(segment, "To Pitch", 0, 100, 600)
    mean_pitch = call(pitch_obj, "Get mean", 0, 0, "Hertz")
    if str(mean_pitch) == '--undefined--':
        mean_pitch = 0

    # Formants
    formant_obj = call(segment, "To Formant (burg)", 0, 5, 5500, 0.025, 50)
    f1 = call(formant_obj, "Get value at time", 1, (start+end)/2, "Hertz", "Linear")
    f2 = call(formant_obj, "Get value at time", 2, (start+end)/2, "Hertz", "Linear")
    if str(f1) == '--undefined--': f1 = 0
    if str(f2) == '--undefined--': f2 = 0

    row = {
        "word": word,
        "start": round(start, 4),
        "end": round(end, 4),
        "duration": round(duration, 4),
        "pitch_mean": round(float(mean_pitch), 2),
        "f1_mean": round(float(f1), 2),
        "f2_mean": round(float(f2), 2),
    }
    for i, val in enumerate(mfcc_means, 1):
        row[f"mfcc_{i}"] = round(float(val), 4)

    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("/Users/karina/Desktop/university/features_ANKA19.csv", index=False)
print(f"Done! Extracted features for {len(df)} words.")
print(df.head())