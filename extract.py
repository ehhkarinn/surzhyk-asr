import parselmouth
import numpy as np
import pandas as pd
import librosa
import textgrid
from parselmouth.praat import call

speakers = ["ANKA19", "BK25", "MAMI21", "ALZA18", "MAAL14","ANKR29", "KAPO24", "TASO15", "VIST09"]

all_rows = []

for speaker in speakers:
    wav_path = f"/Users/karina/Desktop/university/corpus/{speaker}/{speaker}.wav"
    tg_path = f"/Users/karina/Desktop/university/corpus/output/{speaker}/{speaker}.TextGrid"

    try:
        sound = parselmouth.Sound(wav_path)
        tg = textgrid.TextGrid.fromFile(tg_path)
    except Exception as e:
        print(f"Skipping {speaker}: {e}")
        continue

    words_tier = tg[0]

    for interval in words_tier:
        word = interval.mark.strip()
        if word == "" or word == "<unk>" or word == "spn":
            continue

        start = interval.minTime
        end = interval.maxTime
        duration = end - start

        if duration < 0.05:
            continue

        segment = sound.extract_part(from_time=start, to_time=end, preserve_times=False)
        samples = segment.values[0]
        sr = int(segment.sampling_frequency)

        # MFCCs
        n_fft = min(2048, len(samples))
        mfccs = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=13, n_fft=n_fft)
        mfcc_means = np.mean(mfccs, axis=1)

        # Pitch
        try:
            pitch_obj = call(segment, "To Pitch", 0, 100, 600)
            mean_pitch = call(pitch_obj, "Get mean", 0, 0, "Hertz")
            if str(mean_pitch) == '--undefined--':
                mean_pitch = 0
        except:
            mean_pitch = 0

        # Formants
        try:
            formant_obj = call(segment, "To Formant (burg)", 0, 5, 5500, 0.025, 50)
            mid = (start + end) / 2
            f1 = call(formant_obj, "Get value at time", 1, (end-start)/2, "Hertz", "Linear")
            f2 = call(formant_obj, "Get value at time", 2, (end-start)/2, "Hertz", "Linear")
            if str(f1) == '--undefined--': f1 = 0
            if str(f2) == '--undefined--': f2 = 0
        except:
            f1, f2 = 0, 0

        row = {
            "speaker": speaker,
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

        all_rows.append(row)

    print(f"Done: {speaker}")

df = pd.DataFrame(all_rows)
df.to_csv("/Users/karina/Desktop/university/features_all_speakers.csv", index=False)
print(f"\nTotal: {len(df)} words across {len(speakers)} speakers")
print(df.head())