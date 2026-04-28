import os
import json
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import evaluate
import re

wer_metric = evaluate.load("wer")

def normalise(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Load data
dataset_dir = "/Users/karina/Desktop/university/whisper_dataset_9speakers"
with open(os.path.join(dataset_dir, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

chunk_features = pd.read_csv("/Users/karina/Desktop/university/chunk_features.csv")

acoustic_cols = ["mean_pitch", "std_pitch", "mean_f1", "mean_f2"] + \
                [f"mean_mfcc_{i}" for i in range(1, 14)] + \
                [f"std_mfcc_{i}" for i in range(1, 14)]
acoustic_cols = [c for c in acoustic_cols if c in chunk_features.columns]

print(f"Using {len(acoustic_cols)} acoustic features")

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="ukrainian", task="transcribe"
)

# Normalise acoustic features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(chunk_features[acoustic_cols].fillna(0).values)

class SurzhykDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio, sr = sf.read(os.path.join(dataset_dir, "audio", item["file"]))
        audio = audio.astype(np.float32)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        input_features = processor(audio, sampling_rate=16000,
                                   return_tensors="pt").input_features.squeeze(0)
        labels = processor.tokenizer(item["text"],
                                     return_tensors="pt").input_ids.squeeze(0)
        match = chunk_features[chunk_features["file"] == item["file"]]
        if match.empty:
            acoustic = np.zeros(len(acoustic_cols))
        else:
            acoustic = match[acoustic_cols].fillna(0).values[0]
        acoustic = scaler.transform([acoustic])[0]
        acoustic = torch.tensor(acoustic, dtype=torch.float32)
        return input_features, labels, acoustic

split = int(len(metadata) * 0.8)
train_dataset = SurzhykDataset(metadata[:split])
test_dataset  = SurzhykDataset(metadata[split:])

def collate_fn(batch):
    input_features, labels_list, acoustics = zip(*batch)
    input_features = torch.stack(input_features)
    acoustics = torch.stack(acoustics)
    max_len = max(l.size(0) for l in labels_list)
    padded = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
    for i, l in enumerate(labels_list):
        padded[i, :l.size(0)] = l
    return input_features, padded, acoustics

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset,  batch_size=2, shuffle=False, collate_fn=collate_fn)

# Load fine-tuned v3 model as base
print("Loading fine-tuned v3 model...")
model = WhisperForConditionalGeneration.from_pretrained(
    "/Users/karina/Desktop/university/whisper_finetuned_v3/checkpoint-126"
)

# Add small acoustic adapter on top of encoder
n_acoustic = len(acoustic_cols)
d_model = model.config.d_model  # 512

acoustic_adapter = nn.Sequential(
    nn.Linear(n_acoustic, 64),
    nn.Tanh(),
    nn.Linear(64, d_model),
    nn.Tanh()
)

# Freeze everything except last 2 encoder layers + adapter
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last 2 encoder layers
for layer in model.model.encoder.layers[-2:]:
    for param in layer.parameters():
        param.requires_grad = True

forced_ids = processor.get_decoder_prompt_ids(language="ukrainian", task="transcribe")

trainable_params = list(acoustic_adapter.parameters()) + \
                   [p for p in model.parameters() if p.requires_grad]
print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

optimizer = AdamW(trainable_params, lr=5e-6)

best_wer = float("inf")

print("Starting training...\n")
for epoch in range(15):
    # Train
    model.train()
    acoustic_adapter.train()
    total_loss = 0

    for input_features, labels, acoustics in train_loader:
        optimizer.zero_grad()

        # Get encoder output
        enc_out = model.model.encoder(input_features)
        hidden = enc_out.last_hidden_state

        # Add acoustic features as a residual
        a_vec = acoustic_adapter(acoustics).unsqueeze(1)  # (B, 1, d_model)
        hidden = hidden + 0.1 * a_vec  # small residual addition

        enc_out.last_hidden_state = hidden

        out = model(encoder_outputs=enc_out, labels=labels)
        out.loss.backward()
        optimizer.step()
        total_loss += out.loss.item()

    avg_loss = total_loss / len(train_loader)

    # Evaluate
    model.eval()
    acoustic_adapter.eval()
    preds, refs = [], []

    with torch.no_grad():
        for input_features, labels, acoustics in test_loader:
            enc_out = model.model.encoder(input_features)
            hidden = enc_out.last_hidden_state
            a_vec = acoustic_adapter(acoustics).unsqueeze(1)
            hidden = hidden + 0.1 * a_vec
            enc_out.last_hidden_state = hidden

            ids = model.generate(
                encoder_outputs=enc_out,
                forced_decoder_ids=forced_ids
            )
            decoded = processor.batch_decode(ids, skip_special_tokens=True)
            label_ids = labels.clone()
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            ref_str = processor.batch_decode(label_ids, skip_special_tokens=True)
            preds.extend([normalise(p) for p in decoded])
            refs.extend([normalise(r) for r in ref_str])

    wer = wer_metric.compute(predictions=preds, references=refs)
    print(f"Epoch {epoch+1:2d} | loss={avg_loss:.4f} | WER={wer*100:.2f}%")

    if wer < best_wer:
        best_wer = wer
        torch.save(acoustic_adapter.state_dict(),
                   "/Users/karina/Desktop/university/acoustic_adapter_best.pt")
        print(f"           ↑ best adapter saved (WER={best_wer*100:.2f}%)")

print(f"\nBest WER: {best_wer*100:.2f}%")
print(f"Baseline fine-tuned v3 WER was: 4.35%")