import os
import json
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
import re
from sklearn.preprocessing import StandardScaler

wer_metric = evaluate.load("wer")

def normalise(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

dataset_dir = "/Users/karina/Desktop/university/whisper_dataset_unseen3"
with open(os.path.join(dataset_dir, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)


import parselmouth
import textgrid
from parselmouth.praat import call

def extract_features_for_unseen():
    speakers = ["ANKU", "ANMO17", "YULDE10"]
    rows = []
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
            n_fft = min(2048, len(samples))
            mfccs = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=13, n_fft=n_fft)
            mfcc_means = np.mean(mfccs, axis=1)
            try:
                pitch_obj = call(segment, "To Pitch", 0, 100, 600)
                mean_pitch = call(pitch_obj, "Get mean", 0, 0, "Hertz")
                if str(mean_pitch) == '--undefined--': mean_pitch = 0
            except: mean_pitch = 0
            try:
                formant_obj = call(segment, "To Formant (burg)", 0, 5, 5500, 0.025, 50)
                f1 = call(formant_obj, "Get value at time", 1, (end-start)/2, "Hertz", "Linear")
                f2 = call(formant_obj, "Get value at time", 2, (end-start)/2, "Hertz", "Linear")
                if str(f1) == '--undefined--': f1 = 0
                if str(f2) == '--undefined--': f2 = 0
            except: f1, f2 = 0, 0
            row = {"speaker": speaker, "word": word, "start": round(start,4),
                   "end": round(end,4), "duration": round(duration,4),
                   "pitch_mean": round(float(mean_pitch),2),
                   "f1_mean": round(float(f1),2), "f2_mean": round(float(f2),2)}
            for i, val in enumerate(mfcc_means, 1):
                row[f"mfcc_{i}"] = round(float(val), 4)
            rows.append(row)
        print(f"Extracted features for {speaker}")
    return pd.DataFrame(rows)

print("Extracting features for unseen speakers...")
unseen_word_features = extract_features_for_unseen()

acoustic_cols = ["mean_pitch", "std_pitch", "mean_f1", "mean_f2"] + \
                [f"mean_mfcc_{i}" for i in range(1, 14)] + \
                [f"std_mfcc_{i}" for i in range(1, 14)]

chunk_rows = []
for item in metadata:
    speaker = item["speaker"]
    mask = ((unseen_word_features["speaker"] == speaker) &
            (unseen_word_features["start"] >= item["start"]) &
            (unseen_word_features["end"] <= item["end"]))
    words = unseen_word_features[mask]
    if len(words) == 0:
        row = {"file": item["file"]}
        for col in acoustic_cols:
            row[col] = 0.0
    else:
        row = {"file": item["file"],
               "mean_pitch": words["pitch_mean"].mean(),
               "std_pitch": words["pitch_mean"].std(),
               "mean_f1": words["f1_mean"].mean(),
               "mean_f2": words["f2_mean"].mean()}
        for i in range(1, 14):
            col = f"mfcc_{i}"
            if col in words.columns:
                row[f"mean_{col}"] = words[col].mean()
                row[f"std_{col}"] = words[col].std()
    chunk_rows.append(row)

unseen_chunk_features = pd.DataFrame(chunk_rows).fillna(0)
actual_acoustic_cols = [c for c in acoustic_cols if c in unseen_chunk_features.columns]

train_features = pd.read_csv("/Users/karina/Desktop/university/chunk_features.csv")
scaler = StandardScaler()
scaler.fit(train_features[actual_acoustic_cols].fillna(0).values)

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="ukrainian", task="transcribe")
forced_ids = processor.get_decoder_prompt_ids(language="ukrainian", task="transcribe")

base_model = WhisperForConditionalGeneration.from_pretrained(
    "/Users/karina/Desktop/university/whisper_finetuned_v3/checkpoint-126")

acoustic_adapter = nn.Sequential(
    nn.Linear(len(actual_acoustic_cols), 64),
    nn.Tanh(),
    nn.Linear(64, base_model.config.d_model),
    nn.Tanh()
)
acoustic_adapter.load_state_dict(
    torch.load("/Users/karina/Desktop/university/acoustic_adapter_best.pt"))

base_model.eval()
acoustic_adapter.eval()

preds_base, preds_acoustic, refs = [], [], []

for i, item in enumerate(metadata):
    audio, sr = sf.read(os.path.join(dataset_dir, "audio", item["file"]))
    audio = audio.astype(np.float32)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")


    match = unseen_chunk_features[unseen_chunk_features["file"] == item["file"]]
    if match.empty:
        acoustic = np.zeros(len(actual_acoustic_cols))
    else:
        acoustic = match[actual_acoustic_cols].values[0]
    acoustic = scaler.transform([acoustic])[0]
    acoustic_tensor = torch.tensor(acoustic, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
    
        ids_base = base_model.generate(inputs.input_features, forced_decoder_ids=forced_ids)
        pred_base = processor.batch_decode(ids_base, skip_special_tokens=True)[0]

        enc_out = base_model.model.encoder(inputs.input_features)
        hidden = enc_out.last_hidden_state
        a_vec = acoustic_adapter(acoustic_tensor).unsqueeze(1)
        hidden = hidden + 0.1 * a_vec
        enc_out.last_hidden_state = hidden
        ids_acoustic = base_model.generate(encoder_outputs=enc_out, forced_decoder_ids=forced_ids)
        pred_acoustic = processor.batch_decode(ids_acoustic, skip_special_tokens=True)[0]

    preds_base.append(normalise(pred_base))
    preds_acoustic.append(normalise(pred_acoustic))
    refs.append(normalise(item["text"]))

wer_base = wer_metric.compute(predictions=preds_base, references=refs)
wer_acoustic = wer_metric.compute(predictions=preds_acoustic, references=refs)

print("\n=== RESULTS ON UNSEEN SPEAKERS ===")
print(f"Fine-tuned v3 (no acoustic):   {wer_base*100:.2f}%")
print(f"Fine-tuned v3 + acoustic:      {wer_acoustic*100:.2f}%")
diff = wer_base - wer_acoustic
print(f"Difference:                    {diff*100:.2f}% ({'improvement' if diff > 0 else 'degradation'})")
