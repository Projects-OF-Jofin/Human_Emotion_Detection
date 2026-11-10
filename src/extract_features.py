
import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm

DATASET = Path(r"G:\My Drive\Human_Emotion_Detection\dataset")
OUTPUT = Path(r"G:\My Drive\Human_Emotion_Detection\outputs")

OUTPUT.mkdir(parents=True, exist_ok=True)

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

records = []

audio_files = list(DATASET.rglob("*.wav"))

print(f"Extracting features from {len(audio_files)} recordings...")

for file in tqdm(audio_files):
    try:
        filename = file.stem
        parts = filename.split("-")

        emotion_code = parts[2]
        emotion = emotion_map.get(emotion_code)

        if emotion is None:
            continue

        audio, sr = librosa.load(file, sr=22050, duration=4)

        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=40
        )
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=sr
        )
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=64
        )

        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(chroma, axis=1),
            np.mean(mel, axis=1)
        ])

        records.append({
            "filename": filename,
            "emotion": emotion,
            "features": features
        })

    except Exception as e:
        print(f"Error processing {file.name}: {e}")

df = pd.DataFrame(records)

np.save(
    OUTPUT / "X_features.npy",
    np.stack(df["features"].values)
)

df[["filename", "emotion"]].to_csv(
    OUTPUT / "labels.csv", index=False
)

print("\nFEATURE EXTRACTION COMPLETED")
print("Total processed:", len(df))
print("\nEmotion distribution:")
print(df["emotion"].value_counts())
print("Feature shape:", np.load(OUTPUT / "X_features.npy").shape)
