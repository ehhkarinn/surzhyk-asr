import os
import json
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

dataset_dir = "/Users/karina/Desktop/university/whisper_dataset_9speakers"
with open(os.path.join(dataset_dir, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Total chunks: {len(metadata)}")

processor = WhisperProcessor.from_pretrained("openai/whisper-small", 
                                              language="ukrainian", 
                                              task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.eval()

embeddings = []

for item in metadata:
    audio, sr = sf.read(os.path.join(dataset_dir, "audio", item["file"]))
    audio = audio.astype(np.float32)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    
    with torch.no_grad():
        encoder_output = model.model.encoder(inputs.input_features)
        # Mean pool across time dimension to get fixed-size vector
        embedding = encoder_output.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    embeddings.append({
        "file": item["file"],
        "speaker": item["speaker"],
        "text": item["text"],
        "embedding": embedding.tolist()
    })
    print(f"Processed {item['file']} — embedding shape: {embedding.shape}")


np.save("/Users/karina/Desktop/university/whisper_embeddings.npy", 
        np.array([e["embedding"] for e in embeddings]))

with open("/Users/karina/Desktop/university/whisper_embeddings_meta.json", "w") as f:
    json.dump([{k: v for k, v in e.items() if k != "embedding"} for e in embeddings], f, 
              ensure_ascii=False, indent=2)

print(f"\nDone! Embeddings shape: {np.array([e['embedding'] for e in embeddings]).shape}")
print("Saved to whisper_embeddings.npy")
