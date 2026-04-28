import numpy as np
import pandas as pd
import json
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

embeddings = np.load("/Users/karina/Desktop/university/whisper_embeddings.npy")
print(f"Embeddings shape: {embeddings.shape}")

with open("/Users/karina/Desktop/university/whisper_embeddings_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

chunk_features = pd.read_csv("/Users/karina/Desktop/university/chunk_features.csv")
print(f"Chunk features shape: {chunk_features.shape}")

acoustic_cols = ["mean_pitch", "std_pitch", "mean_f1", "mean_f2"] + \
                [f"mean_mfcc_{i}" for i in range(1, 14)] + \
                [f"std_mfcc_{i}" for i in range(1, 14)]

acoustic_cols = [c for c in acoustic_cols if c in chunk_features.columns]

combined_features = []
labels = []
files_matched = []

for i, item in enumerate(meta):
    filename = item["file"]
    speaker = item["speaker"]
    
    match = chunk_features[chunk_features["file"] == filename]
    
    if match.empty:
        print(f"No acoustic features found for {filename}")
        continue
    
    acoustic = match[acoustic_cols].values[0]
    whisper_emb = embeddings[i]
    
    combined = np.concatenate([whisper_emb, acoustic])
    combined_features.append(combined)
    labels.append(speaker)
    files_matched.append(filename)

combined_features = np.array(combined_features)
print(f"\nCombined features shape: {combined_features.shape}")
print(f"Matched {len(files_matched)} chunks")

le = LabelEncoder()
y = le.fit_transform(labels)

scaler = StandardScaler()
X = scaler.fit_transform(combined_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

print("\nTraining MLP classifier...")
clf = MLPClassifier(hidden_layer_sizes=(256, 128), 
                    max_iter=200, 
                    random_state=42,
                    verbose=True)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
