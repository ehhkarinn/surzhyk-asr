import os
import json
import torch
import numpy as np
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
import evaluate

dataset_dir = "/Users/karina/Desktop/university/whisper_dataset_9speakers"
with open(os.path.join(dataset_dir, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

for item in metadata:
    item["audio"] = os.path.join(dataset_dir, "audio", item["file"])

split = int(len(metadata) * 0.8)
train_data = metadata[:split]
test_data = metadata[split:]

print(f"Train: {len(train_data)} chunks, Test: {len(test_data)} chunks")

print("Loading Whisper...")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="ukrainian", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ukrainian", task="transcribe")

def prepare_dataset(items):
    rows = []
    for item in items:
        import soundfile as sf
        audio, sr = sf.read(item["audio"])
        audio = audio.astype(np.float32)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features[0]
        labels = processor.tokenizer(item["text"]).input_ids
        rows.append({
            "input_features": input_features.numpy(),
            "labels": labels
        })
    return rows

print("Preparing training data...")
train_rows = prepare_dataset(train_data)
test_rows = prepare_dataset(test_data)

train_dataset = Dataset.from_list(train_rows)
test_dataset = Dataset.from_list(test_rows)

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollator:
    processor: Any

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollator(processor=processor)

wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

training_args = Seq2SeqTrainingArguments(
    output_dir="/Users/karina/Desktop/university/whisper_finetuned_v3",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=1e-5,
    warmup_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=5,
    predict_with_generate=True,
    generation_max_length=225,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    fp16=False,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

print("Starting fine-tuning...")
trainer.train()

print("Saving model...")
model.save_pretrained("/Users/karina/Desktop/university/whisper_finetuned_v3/final")
processor.save_pretrained("/Users/karina/Desktop/university/whisper_finetuned_v3/final")
print("Done!")
