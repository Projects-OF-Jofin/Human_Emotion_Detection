from pathlib import Path
import json
import warnings
import time

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "dataset"
OUTPUT = BASE / "outputs"
MODELS = BASE / "models"
GRAPHS = OUTPUT / "graphs"

for folder in [OUTPUT, MODELS, GRAPHS]:
    folder.mkdir(parents=True, exist_ok=True)

MODEL_ID = "facebook/wav2vec2-base"

SAMPLE_RATE = 16000
MAX_DURATION = 4.0

EXTRACTION_BATCH_SIZE = 4
TRAIN_BATCH_SIZE = 32

EPOCHS = 40
LEARNING_RATE = 0.0005
PATIENCE = 7

RANDOM_STATE = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

print("=" * 65)
print("WAV2VEC 2.0 - SPEECH EMOTION RECOGNITION")
print("=" * 65)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2
        ),
        "GB"
    )

# ==========================================================
# 2. DATASET DISCOVERY
# ==========================================================

def discover_audio_files():

    files = sorted(
        path for path in DATASET.rglob("*.wav")
        if path.parent.name.startswith("Actor_")
    )

    if not files:
        raise FileNotFoundError(
            f"No RAVDESS WAV files found inside {DATASET}"
        )

    if len(files) != 1440:
        print(
            f"WARNING: Expected 1440 recordings, found {len(files)}."
        )

    return files


# ==========================================================
# 3. EXTRACT LABELS AND ACTOR IDS
# ==========================================================

def get_metadata(path):

    parts = path.stem.split("-")

    if len(parts) != 7:
        raise ValueError(
            f"Invalid RAVDESS filename: {path.name}"
        )

    emotion_code = parts[2]
    actor_id = parts[6]

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

    if emotion_code not in emotion_map:
        raise ValueError(
            f"Unknown emotion code in {path.name}"
        )

    return emotion_map[emotion_code], actor_id


# ==========================================================
# 4. LOAD PRETRAINED WAV2VEC 2.0
# ==========================================================

print("\nLoading pretrained Wav2Vec 2.0...")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    MODEL_ID
)

wav2vec_model = Wav2Vec2Model.from_pretrained(
    MODEL_ID
)

wav2vec_model = wav2vec_model.to(DEVICE)
wav2vec_model.eval()

# Freeze all Transformer parameters.
for parameter in wav2vec_model.parameters():
    parameter.requires_grad = False

print("Pretrained model loaded:", MODEL_ID)
print("Hidden dimension:", wav2vec_model.config.hidden_size)


# ==========================================================
# 5. EXTRACT AND CACHE AUDIO EMBEDDINGS
# ==========================================================

def extract_embeddings():

    cache_file = OUTPUT / "wav2vec2_embeddings.npz"

    if cache_file.exists():

        print("\nLoading cached Wav2Vec embeddings...")

        cached = np.load(cache_file, allow_pickle=False)

        embeddings = cached["embeddings"].astype(np.float32)
        labels = cached["labels"].astype(str)
        groups = cached["groups"].astype(str)
        filenames = cached["filenames"].astype(str)

        print("Cached embeddings:", embeddings.shape)

        return embeddings, labels, groups, filenames

    files = discover_audio_files()

    print("\nAudio recordings discovered:", len(files))
    print("Extracting Wav2Vec embeddings...")

    all_embeddings = []
    all_labels = []
    all_groups = []
    all_filenames = []

    for start in range(0, len(files), EXTRACTION_BATCH_SIZE):

        batch_files = files[
            start:start + EXTRACTION_BATCH_SIZE
        ]

        batch_audio = []
        valid_files = []

        for path in batch_files:

            try:

                audio, _ = librosa.load(
                    path,
                    sr=SAMPLE_RATE,
                    duration=MAX_DURATION,
                    mono=True
                )

                if len(audio) == 0:
                    continue

                batch_audio.append(audio.astype(np.float32))
                valid_files.append(path)

            except Exception as error:

                print(
                    f"Skipping {path.name}: {error}"
                )

        if not batch_audio:
            continue

        inputs = feature_extractor(
            batch_audio,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        input_values = inputs["input_values"].to(DEVICE)

        attention_mask = inputs.get("attention_mask")

        if attention_mask is not None:
            attention_mask = attention_mask.to(DEVICE)

        with torch.inference_mode():

            outputs = wav2vec_model(
                input_values=input_values,
                attention_mask=attention_mask
            )

            hidden = outputs.last_hidden_state

            # Mean pooling over valid speech frames.
            if attention_mask is not None:

                feature_mask = (
                    wav2vec_model._get_feature_vector_attention_mask(
                        hidden.shape[1],
                        attention_mask
                    )
                )

                feature_mask = feature_mask.unsqueeze(-1).to(
                    hidden.dtype
                )

                pooled = (
                    (hidden * feature_mask).sum(dim=1)
                    /
                    feature_mask.sum(dim=1).clamp(min=1)
                )

            else:

                pooled = hidden.mean(dim=1)

            pooled = pooled.cpu().numpy().astype(np.float32)

        for i, path in enumerate(valid_files):

            emotion, actor = get_metadata(path)

            all_embeddings.append(pooled[i])
            all_labels.append(emotion)
            all_groups.append(actor)
            all_filenames.append(path.name)

        completed = min(start + len(batch_files), len(files))

        print(
            f"\rProcessed {completed}/{len(files)} recordings",
            end=""
        )

    print("\nEmbedding extraction completed.")

    embeddings = np.asarray(
        all_embeddings,
        dtype=np.float32
    )

    labels = np.asarray(all_labels, dtype=str)
    groups = np.asarray(all_groups, dtype=str)
    filenames = np.asarray(all_filenames, dtype=str)

    if len(embeddings) == 0:
        raise RuntimeError("No audio embeddings were extracted.")

    np.savez_compressed(
        cache_file,
        embeddings=embeddings,
        labels=labels,
        groups=groups,
        filenames=filenames
    )

    print("Embedding shape:", embeddings.shape)
    print("Cache saved:", cache_file)

    return embeddings, labels, groups, filenames


# ==========================================================
# 6. LOAD EMBEDDINGS
# ==========================================================

X, y_text, groups, filenames = extract_embeddings()

print("\nDataset shape:", X.shape)
print("Emotion classes:", np.unique(y_text))
print("Unique actors:", len(np.unique(groups)))

print("\nEmotion distribution:")
print(pd.Series(y_text).value_counts())

# ==========================================================
# 7. ENCODE LABELS
# ==========================================================

encoder = LabelEncoder()

y = encoder.fit_transform(y_text)

classes = encoder.classes_
num_classes = len(classes)

if num_classes != 8:
    raise ValueError(
        f"Expected 8 emotion classes, found {num_classes}"
    )

# ==========================================================
# 8. ACTOR-INDEPENDENT TRAIN / VALIDATION / TEST SPLIT
# ==========================================================

print("\nCreating actor-independent data splits...")

test_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

trainval_idx, test_idx = next(
    test_splitter.split(X, y, groups)
)

val_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

relative_train_idx, relative_val_idx = next(
    val_splitter.split(
        X[trainval_idx],
        y[trainval_idx],
        groups[trainval_idx]
    )
)

train_idx = trainval_idx[relative_train_idx]
val_idx = trainval_idx[relative_val_idx]

# Verify no speaker overlap.
train_actors = set(groups[train_idx])
val_actors = set(groups[val_idx])
test_actors = set(groups[test_idx])

assert not train_actors.intersection(val_actors)
assert not train_actors.intersection(test_actors)
assert not val_actors.intersection(test_actors)

print("Training recordings:", len(train_idx))
print("Validation recordings:", len(val_idx))
print("Testing recordings:", len(test_idx))

print("Training actors:", sorted(train_actors))
print("Validation actors:", sorted(val_actors))
print("Testing actors:", sorted(test_actors))

# ==========================================================
# 9. NORMALIZE EMBEDDINGS USING TRAINING DATA ONLY
# ==========================================================

mean = X[train_idx].mean(axis=0)
std = X[train_idx].std(axis=0)

std[std < 1e-8] = 1.0

X_normalized = (X - mean) / std

np.savez(
    OUTPUT / "wav2vec2_normalization.npz",
    mean=mean,
    std=std
)

# ==========================================================
# 10. CREATE PYTORCH DATA LOADERS
# ==========================================================

def make_loader(indices, shuffle=False):

    features = torch.tensor(
        X_normalized[indices],
        dtype=torch.float32
    )

    targets = torch.tensor(
        y[indices],
        dtype=torch.long
    )

    dataset = TensorDataset(features, targets)

    return DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )


train_loader = make_loader(train_idx, shuffle=True)
val_loader = make_loader(val_idx)
test_loader = make_loader(test_idx)

# ==========================================================
# 11. CLASSIFICATION HEAD
# ==========================================================

class EmotionClassifier(nn.Module):

    def __init__(self, input_dim, output_dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, output_dim)
        )

    def forward(self, x):

        return self.network(x)


classifier = EmotionClassifier(
    input_dim=X.shape[1],
    output_dim=num_classes
).to(DEVICE)

# ==========================================================
# 12. CLASS WEIGHTS
# ==========================================================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y[train_idx]),
    y=y[train_idx]
)

weight_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=weight_tensor
)

optimizer = torch.optim.AdamW(
    classifier.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)

# ==========================================================
# 13. EVALUATION FUNCTION
# ==========================================================

def evaluate(loader):

    classifier.eval()

    actual = []
    predicted = []

    total_loss = 0.0

    with torch.inference_mode():

        for features, targets in loader:

            features = features.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = classifier(features)

            loss = criterion(logits, targets)

            total_loss += loss.item() * len(targets)

            predictions = torch.argmax(
                logits,
                dim=1
            )

            actual.extend(targets.cpu().numpy())
            predicted.extend(predictions.cpu().numpy())

    metrics = {
        "loss": total_loss / len(actual),
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(
            actual,
            predicted,
            average="macro",
            zero_division=0
        )
    }

    return metrics, np.asarray(actual), np.asarray(predicted)


# ==========================================================
# 14. TRAIN CLASSIFIER
# ==========================================================

print("\n" + "=" * 65)
print("TRAINING WAV2VEC 2.0 CLASSIFICATION HEAD")
print("=" * 65)

best_val_f1 = -1.0
best_epoch = 0
patience_counter = 0

history = {
    "train_loss": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_macro_f1": []
}

checkpoint_path = MODELS / "wav2vec2_best_head.pt"

start_time = time.time()

for epoch in range(1, EPOCHS + 1):

    classifier.train()

    running_loss = 0.0
    samples = 0

    for features, targets in train_loader:

        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        logits = classifier(features)

        loss = criterion(logits, targets)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * len(targets)
        samples += len(targets)

    train_loss = running_loss / samples

    val_metrics, _, _ = evaluate(val_loader)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_metrics["loss"])
    history["val_accuracy"].append(val_metrics["accuracy"])
    history["val_macro_f1"].append(val_metrics["macro_f1"])

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val Accuracy: {val_metrics['accuracy']:.4f} | "
        f"Val Macro F1: {val_metrics['macro_f1']:.4f}"
    )

    if val_metrics["macro_f1"] > best_val_f1:

        best_val_f1 = val_metrics["macro_f1"]
        best_epoch = epoch
        patience_counter = 0

        torch.save(
            classifier.state_dict(),
            checkpoint_path
        )

    else:

        patience_counter += 1

    if patience_counter >= PATIENCE:

        print("\nEarly stopping triggered.")
        break

training_time = (time.time() - start_time) / 60

print("\nBest epoch:", best_epoch)
print("Best validation Macro F1:", best_val_f1)
print("Training time:", round(training_time, 2), "minutes")

# ==========================================================
# 15. LOAD BEST CHECKPOINT
# ==========================================================

classifier.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )
)

classifier.eval()

# ==========================================================
# 16. FINAL TEST EVALUATION
# ==========================================================

print("\n" + "=" * 65)
print("FINAL TEST EVALUATION")
print("=" * 65)

test_metrics, y_true, y_pred = evaluate(test_loader)

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    target_names=classes,
    zero_division=0
)

print(report)

# ==========================================================
# 17. SAVE RESULTS
# ==========================================================

results = {
    "Model": "Wav2Vec2_Frozen_Classifier",
    "Accuracy": float(accuracy),
    "Precision": float(precision),
    "Recall": float(recall),
    "Macro_F1": float(macro_f1),
    "Best_Validation_F1": float(best_val_f1),
    "Best_Epoch": int(best_epoch),
    "Training_Time_Minutes": float(training_time),
    "Test_Actors": sorted(test_actors)
}

with open(
    OUTPUT / "wav2vec2_results.json",
    "w"
) as file:

    json.dump(results, file, indent=4)

pd.DataFrame([results]).to_csv(
    OUTPUT / "wav2vec2_results.csv",
    index=False
)

with open(
    OUTPUT / "wav2vec2_classification_report.txt",
    "w"
) as file:

    file.write(report)

# ==========================================================
# 18. SAVE CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(num_classes)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

fig, ax = plt.subplots(figsize=(10, 8))

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    colorbar=False
)

plt.title("Wav2Vec 2.0 - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    GRAPHS / "wav2vec2_confusion_matrix.png",
    dpi=300
)

plt.close()

# ==========================================================
# 19. SAVE TRAINING CURVES
# ==========================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(
    history["train_loss"],
    label="Training Loss"
)

axes[0].plot(
    history["val_loss"],
    label="Validation Loss"
)

axes[0].set_title("Training and Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(
    history["val_accuracy"],
    label="Validation Accuracy"
)

axes[1].plot(
    history["val_macro_f1"],
    label="Validation Macro F1"
)

axes[1].set_title("Validation Performance")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Score")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

plt.savefig(
    GRAPHS / "wav2vec2_training_curves.png",
    dpi=300
)

plt.close()

# ==========================================================
# 20. SAVE MODEL METADATA
# ==========================================================

metadata = {
    "model_id": MODEL_ID,
    "sample_rate": SAMPLE_RATE,
    "max_duration": MAX_DURATION,
    "classes": classes.tolist(),
    "embedding_dimension": int(X.shape[1]),
    "best_epoch": int(best_epoch),
    "test_accuracy": float(accuracy),
    "test_macro_f1": float(macro_f1)
}

with open(
    MODELS / "wav2vec2_metadata.json",
    "w"
) as file:

    json.dump(metadata, file, indent=4)

# ==========================================================
# 21. FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 65)
print("WAV2VEC 2.0 EXPERIMENT COMPLETED")
print("=" * 65)

print("Best epoch:", best_epoch)
print(f"Test accuracy: {accuracy:.2%}")
print(f"Test Macro F1: {macro_f1:.4f}")

print("\nSaved outputs:", OUTPUT)
print("Saved model:", checkpoint_path)
print("Saved graphs:", GRAPHS)

print("\nALL DONE!")