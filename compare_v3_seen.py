import os
import json
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate

wer_metric = evaluate.load("wer")

dataset_dir = "/Users/karina/Desktop/university/whisper_dataset_9speakers"
with open(os.path.join(dataset_dir, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Use last 20% as test set — same split as training
split = int(len(metadata) * 0.8)
test_data = metadata[split:]
print(f"Test chunks: {len(test_data)}")

import re

def normalise(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def transcribe(model, processor, test_data):
    predictions = []
    references = []
    for item in test_data:
        audio, sr = sf.read(os.path.join(dataset_dir, "audio", item["file"]))
        audio = audio.astype(np.float32)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            predicted_ids = model.generate(inputs.input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print(f"REF:  {normalise(item['text'])}")
        print(f"PRED: {normalise(transcription)}")
        print()
        predictions.append(transcription)
        references.append(item["text"])
    predictions_norm = [normalise(p) for p in predictions]
    references_norm = [normalise(r) for r in references]
    wer = wer_metric.compute(predictions=predictions_norm, references=references_norm)
    return wer, predictions, references

print("=== BASELINE WHISPER ===")
processor_base = WhisperProcessor.from_pretrained("openai/whisper-small", language="ukrainian", task="transcribe")
model_base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model_base.config.forced_decoder_ids = processor_base.get_decoder_prompt_ids(language="ukrainian", task="transcribe")
wer_base, _, _ = transcribe(model_base, processor_base, test_data)
print(f"Baseline WER: {wer_base*100:.2f}%")

print("=== FINE-TUNED WHISPER V3 ===")
processor_ft = WhisperProcessor.from_pretrained("openai/whisper-small", language="ukrainian", task="transcribe")
model_ft = WhisperForConditionalGeneration.from_pretrained("/Users/karina/Desktop/university/whisper_finetuned_v3/checkpoint-126")
model_ft.config.forced_decoder_ids = processor_ft.get_decoder_prompt_ids(language="ukrainian", task="transcribe")
wer_ft, _, _ = transcribe(model_ft, processor_ft, test_data)
print(f"Fine-tuned WER: {wer_ft*100:.2f}%")

print("\n=== SUMMARY ===")
print(f"Baseline WER:   {wer_base*100:.2f}%")
print(f"Fine-tuned WER: {wer_ft*100:.2f}%")
improvement = (wer_base - wer_ft) / wer_base * 100
print(f"Improvement:    {improvement:.2f}%")