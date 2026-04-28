import os
import json
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import evaluate
import re

wer_metric = evaluate.load("wer")

def normalise(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

dataset_dir = "/Users/karina/Desktop/university/whisper_dataset_9speakers"
with open(os.path.join(dataset_dir, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

chunk_features = pd.read_csv("/Users/karina/Desktop/university/chunk_features.csv")

acoustic_cols = ["mean_pitch", "std_pitch", "mean_f1", "mean_f2"] + \
                [f"mean_mfcc_{i}" for i in range(1, 14)] + \
                [f"std_mfcc_{i}" for i in range(1, 14)]
acoustic_cols = [c for c in acoustic_cols if c in chunk_features.columns]

print(f"Using {len(acoustic_cols)} acoustic features")
print(f"Total chunks: {len(metadata)}")

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="ukrainian", task="transcribe"
)

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
            acoustic = torch.zeros(len(acoustic_cols))
        else:
            vals = match[acoustic_cols].values[0]
            vals = np.nan_to_num(vals, nan=0.0)
            acoustic = torch.tensor(vals, dtype=torch.float32)

        return input_features, labels, acoustic

split = int(len(metadata) * 0.8)
train_data = metadata[:split]
test_data  = metadata[split:]
print(f"Train: {len(train_data)}, Test: {len(test_data)}")

train_dataset = SurzhykDataset(train_data)
test_dataset  = SurzhykDataset(test_data)

def collate_fn(batch):
    input_features, labels_list, acoustics = zip(*batch)
    input_features = torch.stack(input_features)
    acoustics      = torch.stack(acoustics)
    max_len = max(l.size(0) for l in labels_list)
    padded  = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
    for i, l in enumerate(labels_list):
        padded[i, :l.size(0)] = l
    return input_features, padded, acoustics

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,
                          collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset,  batch_size=2, shuffle=False,
                          collate_fn=collate_fn)

class AcousticWhisper(nn.Module):
    def __init__(self, n_acoustic):
        super().__init__()
        self.whisper = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small"
        )
        d_model = self.whisper.config.d_model   

        self.acoustic_proj = nn.Sequential(
            nn.Linear(n_acoustic, 128),
            nn.ReLU(),
            nn.Linear(128, d_model),
        )

        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, input_features, labels, acoustic_feats):

        enc_out = self.whisper.model.encoder(input_features)
        hidden  = enc_out.last_hidden_state          

        a_emb = self.acoustic_proj(acoustic_feats).unsqueeze(1)  
        a_emb = a_emb.expand_as(hidden)                          

        fused = self.gate(torch.cat([hidden, a_emb], dim=-1))   
        fused = torch.tanh(fused)

        enc_out.last_hidden_state = fused

        out = self.whisper(
            encoder_outputs=enc_out,
            labels=labels,
        )
        return out.loss

    def generate(self, input_features, acoustic_feats, forced_decoder_ids):
        enc_out = self.whisper.model.encoder(input_features)
        hidden  = enc_out.last_hidden_state

        a_emb = self.acoustic_proj(acoustic_feats).unsqueeze(1).expand_as(hidden)
        fused = torch.tanh(self.gate(torch.cat([hidden, a_emb], dim=-1)))
        enc_out.last_hidden_state = fused

        return self.whisper.generate(
            encoder_outputs=enc_out,
            forced_decoder_ids=forced_decoder_ids,
        )

n_acoustic = len(acoustic_cols)
model = AcousticWhisper(n_acoustic)

for param in model.whisper.parameters():
    param.requires_grad = False

trainable = [p for p in model.parameters() if p.requires_grad]
print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")
optimizer = AdamW(trainable, lr=1e-4)

forced_ids = processor.get_decoder_prompt_ids(language="ukrainian", task="transcribe")

print(f"\nModel has {sum(p.numel() for p in model.parameters()):,} parameters")
print("Starting training...\n")

best_wer = float("inf")
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for input_features, labels, acoustics in train_loader:
        optimizer.zero_grad()
        loss = model(input_features, labels, acoustics)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)


    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for input_features, labels, acoustics in test_loader:
            ids = model.generate(input_features, acoustics, forced_ids)
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
        torch.save(model.state_dict(),
                   "/Users/karina/Desktop/university/acoustic_whisper_best.pt")
        print(f"           ↑ best model saved (WER={best_wer*100:.2f}%)")

print(f"\nBest WER: {best_wer*100:.2f}%")
