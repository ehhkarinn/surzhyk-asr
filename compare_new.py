import os
import json
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate

wer_metric = evaluate.load("wer")

dataset_dir = "/Users/karina/Desktop/university/whisper_dataset_new"
with open(os.path.join(dataset_dir, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Only YULDE10
metadata = [item for item in metadata if item["speaker"] == "YULDE10"]
print(f"Testing on {len(metadata)} chunks from YULDE10")

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
        predictions.append(transcription)
        references.append(item["text"])
    wer = wer_metric.compute(predictions=predictions, references=references)
    return wer

print("=== BASELINE WHISPER ===")
processor_base = WhisperProcessor.from_pretrained("openai/whisper-small", language="ukrainian", task="transcribe")
model_base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model_base.config.forced_decoder_ids = processor_base.get_decoder_prompt_ids(language="ukrainian", task="transcribe")
wer_base = transcribe(model_base, processor_base, metadata)
print(f"Baseline WER: {wer_base*100:.2f}%")

print("=== FINE-TUNED WHISPER V2 ===")
processor_ft = WhisperProcessor.from_pretrained("/Users/karina/Desktop/university/whisper_finetuned_v2/final", language="ukrainian", task="transcribe")
model_ft = WhisperForConditionalGeneration.from_pretrained("/Users/karina/Desktop/university/whisper_finetuned_v2/final")
model_ft.config.forced_decoder_ids = processor_ft.get_decoder_prompt_ids(language="ukrainian", task="transcribe")
wer_ft = transcribe(model_ft, processor_ft, metadata)
print(f"Fine-tuned WER: {wer_ft*100:.2f}%")

print("\n=== SUMMARY ===")
print(f"Baseline WER:   {wer_base*100:.2f}%")
print(f"Fine-tuned WER: {wer_ft*100:.2f}%")
improvement = (wer_base - wer_ft) / wer_base * 100
print(f"Improvement:    {improvement:.2f}%")