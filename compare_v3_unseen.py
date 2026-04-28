import os
import json
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate

wer_metric = evaluate.load("wer")

dataset_dir = "/Users/karina/Desktop/university/whisper_dataset_unseen3"
with open(os.path.join(dataset_dir, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Total test chunks: {len(metadata)}")
print(f"Speakers: {set(item['speaker'] for item in metadata)}\n")

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

# Baseline
print("=== BASELINE WHISPER ===")
processor_base = WhisperProcessor.from_pretrained("openai/whisper-small", language="ukrainian", task="transcribe")
model_base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model_base.config.forced_decoder_ids = processor_base.get_decoder_prompt_ids(language="ukrainian", task="transcribe")
wer_base_all, _, _ = transcribe(model_base, processor_base, metadata)
print(f"Overall Baseline WER: {wer_base_all*100:.2f}%")

# Per speaker baseline
for speaker in ["ANKU", "ANMO17", "YULDE10"]:
    speaker_data = [item for item in metadata if item["speaker"] == speaker]
    wer, _, _ = transcribe(model_base, processor_base, speaker_data)
    print(f"  {speaker}: {wer*100:.2f}%")

# Fine-tuned v3
print("\n=== FINE-TUNED WHISPER V3 ===")
processor_ft = WhisperProcessor.from_pretrained("openai/whisper-small", language="ukrainian", task="transcribe")
model_ft = WhisperForConditionalGeneration.from_pretrained("/Users/karina/Desktop/university/whisper_finetuned_v3/checkpoint-126")
model_ft.config.forced_decoder_ids = processor_ft.get_decoder_prompt_ids(language="ukrainian", task="transcribe")
wer_ft_all, _, _ = transcribe(model_ft, processor_ft, metadata)
print(f"Overall Fine-tuned WER: {wer_ft_all*100:.2f}%")

# Per speaker fine-tuned
for speaker in ["ANKU", "ANMO17", "YULDE10"]:
    speaker_data = [item for item in metadata if item["speaker"] == speaker]
    wer, _, _ = transcribe(model_ft, processor_ft, speaker_data)
    print(f"  {speaker}: {wer*100:.2f}%")

print("\n=== SUMMARY ===")
print(f"Baseline WER (all unseen):   {wer_base_all*100:.2f}%")
print(f"Fine-tuned WER (all unseen): {wer_ft_all*100:.2f}%")
improvement = (wer_base_all - wer_ft_all) / wer_base_all * 100
print(f"Improvement:                 {improvement:.2f}%")

results = {
    "baseline_wer": wer_base_all,
    "finetuned_wer": wer_ft_all,
    "improvement": improvement,
    "test_speakers": ["ANKU", "ANMO17", "YULDE10"],
    "test_chunks": len(metadata)
}
with open("/Users/karina/Desktop/university/evaluation_v3_unseen.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to evaluation_v3_unseen.json")