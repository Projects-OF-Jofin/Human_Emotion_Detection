
from pathlib import Path
import json
import random
import time
import warnings

import numpy as np
import pandas as pd
import librosa
import joblib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# =========================================================
# 1. CONFIGURATION
# =========================================================

SEED = 42
SAMPLE_RATE = 22050
DURATION = 4
N_MFCC = 40
N_MELS = 64
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 6
LEARNING_RATE = 0.001

BASE = Path(__file__).resolve().parent.parent

DATASET = BASE / "dataset" / ""
OUTPUT = BASE / "outputs"
MODELS = BASE / "models"
CHECKPOINTS = MODELS / "checkpoints"
GRAPHS = OUTPUT / "graphs"

for folder in [OUTPUT, MODELS, CHECKPOINTS, GRAPHS]:
    folder.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

print("=" * 65)
print("HUMAN EMOTION DETECTION - AUTOMATED DEEP LEARNING")
print("=" * 65)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory:", round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
    ), "GB")

# =========================================================
# 2. DATASET CONFIGURATION
# =========================================================

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# =========================================================
# 3. FEATURE EXTRACTION AND CACHING
# =========================================================

CACHE_FILE = OUTPUT / "dl_audio_features.npz"


def extract_features():

    if CACHE_FILE.exists():
        print("\nLoading cached DL features...")
        cached = np.load(CACHE_FILE)

        return (
            cached["mfcc"],
            cached["mel"],
            cached["labels"],
            cached["groups"]
        )

    print("\nExtracting MFCC and Mel spectrogram sequences...")

    files = sorted(DATASET.glob("Actor_*/*.wav"))

    if not files:
        raise FileNotFoundError(
            f"No WAV files found in {DATASET}"
        )

    mfcc_features = []
    mel_features = []
    labels = []
    groups = []

    for i, file in enumerate(files, start=1):

        parts = file.stem.split("-")

        emotion_code = parts[2]
        actor_id = parts[-1]

        if emotion_code not in EMOTION_MAP:
            continue

        audio, sr = librosa.load(
            file,
            sr=SAMPLE_RATE,
            duration=DURATION
        )

        audio = librosa.util.fix_length(
            audio,
            size=SAMPLE_RATE * DURATION
        )

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=2048,
            hop_length=512
        )

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=N_MELS
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        mfcc_features.append(mfcc.astype(np.float32))
        mel_features.append(mel_db.astype(np.float32))

        labels.append(EMOTION_MAP[emotion_code])
        groups.append(actor_id)

        if i % 100 == 0:
            print(f"Processed {i}/{len(files)} audio files")

    mfcc_features = np.asarray(mfcc_features, dtype=np.float32)
    mel_features = np.asarray(mel_features, dtype=np.float32)
    labels = np.asarray(labels, dtype=str)
    groups = np.asarray(groups, dtype=str)

    np.savez_compressed(
        CACHE_FILE,
        mfcc=mfcc_features,
        mel=mel_features,
        labels=labels,
        groups=groups
    )

    print("\nFeature extraction completed.")
    print("MFCC shape:", mfcc_features.shape)
    print("Mel shape:", mel_features.shape)

    return mfcc_features, mel_features, labels, groups


# =========================================================
# 4. LOAD AND SPLIT DATA
# =========================================================

mfcc, mel, labels, groups = extract_features()

encoder = LabelEncoder()
y = encoder.fit_transform(labels)

NUM_CLASSES = len(encoder.classes_)

print("\nClasses:", list(encoder.classes_))
print("Total samples:", len(y))
print("Total actors:", len(np.unique(groups)))

# First split: 80% train/validation, 20% test actors
split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED
)

train_val_idx, test_idx = next(
    split1.split(mfcc, y, groups)
)

# Second split: validation actors from training actors
split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED
)

relative_train_idx, relative_val_idx = next(
    split2.split(
        mfcc[train_val_idx],
        y[train_val_idx],
        groups[train_val_idx]
    )
)

train_idx = train_val_idx[relative_train_idx]
val_idx = train_val_idx[relative_val_idx]

print("\nTRAINING SAMPLES:", len(train_idx))
print("VALIDATION SAMPLES:", len(val_idx))
print("TEST SAMPLES:", len(test_idx))

print("\nActors:")
print("Train:", len(np.unique(groups[train_idx])))
print("Validation:", len(np.unique(groups[val_idx])))
print("Test:", len(np.unique(groups[test_idx])))

# =========================================================
# 5. NORMALIZE FEATURES USING TRAINING DATA ONLY
# =========================================================

# MFCC normalization across training samples and time
mfcc_mean = mfcc[train_idx].mean(axis=(0, 2), keepdims=True)
mfcc_std = mfcc[train_idx].std(axis=(0, 2), keepdims=True) + 1e-6

mfcc = (mfcc - mfcc_mean) / mfcc_std

# Per-recording Mel normalization
mel_mean = mel.mean(axis=(1, 2), keepdims=True)
mel_std = mel.std(axis=(1, 2), keepdims=True) + 1e-6

mel = (mel - mel_mean) / mel_std

# Save normalization parameters
np.savez(
    OUTPUT / "dl_normalization.npz",
    mfcc_mean=mfcc_mean,
    mfcc_std=mfcc_std
)

# =========================================================
# 6. PYTORCH DATASETS
# =========================================================

def make_loader(features, indices, batch_size=BATCH_SIZE, shuffle=False):

    x = torch.tensor(
        features[indices],
        dtype=torch.float32
    )

    target = torch.tensor(
        y[indices],
        dtype=torch.long
    )

    dataset = TensorDataset(x, target)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )


# =========================================================
# 7. MODEL ARCHITECTURES
# =========================================================

class CNN1D(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(40, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            nn.Dropout(0.4),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        return self.network(x)


class CNN2D(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Dropout(0.4),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        return self.network(x)


class CNNLSTM(nn.Module):

    def __init__(self, bidirectional=False):
        super().__init__()

        self.bidirectional = bidirectional

        self.conv = nn.Sequential(
            nn.Conv1d(40, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )

        output_size = 128 if bidirectional else 64

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(output_size, NUM_CLASSES)
        )

    def forward(self, x):

        x = self.conv(x)
        x = x.transpose(1, 2)

        output, _ = self.lstm(x)

        # Last temporal output
        output = output[:, -1, :]

        return self.classifier(output)


# =========================================================
# 8. TRAINING FUNCTION
# =========================================================

def train_model(name, model, train_loader, val_loader, test_loader):

    print("\n" + "=" * 65)
    print("STARTING:", name)
    print("=" * 65)

    model = model.to(DEVICE)

    classes = np.unique(y[train_idx])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y[train_idx]
    )

    class_weights = torch.ones(
        NUM_CLASSES,
        dtype=torch.float32
    )

    for cls, weight in zip(classes, weights):
        class_weights[cls] = weight

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": []
    }

    best_path = CHECKPOINTS / f"{name}_best.pt"
    last_path = CHECKPOINTS / f"{name}_last.pt"

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(
                DEVICE,
                non_blocking=True
            )

            batch_y = batch_y.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)

            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()

        val_loss_total = 0.0
        val_predictions = []
        val_targets = []

        with torch.no_grad():

            for batch_x, batch_y in val_loader:

                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                outputs = model(batch_x)

                loss = criterion(outputs, batch_y)

                val_loss_total += loss.item() * batch_x.size(0)

                predictions = outputs.argmax(dim=1)

                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        val_loss = val_loss_total / len(val_loader.dataset)

        val_accuracy = accuracy_score(
            val_targets,
            val_predictions
        )

        val_f1 = f1_score(
            val_targets,
            val_predictions,
            average="macro",
            zero_division=0
        )

        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_f1"].append(val_f1)

        print(
            f"{name} | Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        # Save latest checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_val_f1,
                "history": history
            },
            last_path
        )

        # Save best checkpoint
        if val_f1 > best_val_f1:

            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                model.state_dict(),
                best_path
            )

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    # =====================================================
    # 9. TEST BEST CHECKPOINT
    # =====================================================

    model.load_state_dict(
        torch.load(
            best_path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()

    test_predictions = []
    test_targets = []

    with torch.no_grad():

        for batch_x, batch_y in test_loader:

            batch_x = batch_x.to(DEVICE)

            outputs = model(batch_x)

            predictions = outputs.argmax(dim=1)

            test_predictions.extend(predictions.cpu().numpy())
            test_targets.extend(batch_y.numpy())

    accuracy = accuracy_score(
        test_targets,
        test_predictions
    )

    precision = precision_score(
        test_targets,
        test_predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        test_targets,
        test_predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        test_targets,
        test_predictions,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        test_targets,
        test_predictions,
        labels=np.arange(NUM_CLASSES),
        target_names=encoder.classes_,
        zero_division=0
    )

    print("\nTEST RESULTS:", name)
    print(report)

    # Save classification report
    (OUTPUT / f"{name}_report.txt").write_text(report)

    # Confusion matrix
    cm = confusion_matrix(
        test_targets,
        test_predictions,
        labels=np.arange(NUM_CLASSES)
    )

    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{name} - Confusion Matrix")
    plt.colorbar()

    plt.xticks(
        np.arange(NUM_CLASSES),
        encoder.classes_,
        rotation=45
    )

    plt.yticks(
        np.arange(NUM_CLASSES),
        encoder.classes_
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig(
        GRAPHS / f"{name}_confusion_matrix.png",
        dpi=150
    )

    plt.close()

    # Training curves
    plt.figure(figsize=(10, 5))

    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")

    plt.title(f"{name} - Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        GRAPHS / f"{name}_loss.png",
        dpi=150
    )

    plt.close()

    result = {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Macro_F1": macro_f1,
        "Best_Validation_F1": best_val_f1,
        "Best_Epoch": best_epoch,
        "Training_Time_Minutes": round(
            (time.time() - start_time) / 60, 2
        )
    }

    print("\nCompleted:", name)
    print(json.dumps(result, indent=2))

    return result


# =========================================================
# 10. DEFINE EXPERIMENTS
# =========================================================

experiments = [
    {
        "name": "CNN_1D_MFCC",
        "model": CNN1D,
        "features": mfcc,
        "channel": False
    },
    {
        "name": "CNN_2D_MEL",
        "model": CNN2D,
        "features": mel[:, np.newaxis, :, :],
        "channel": True
    },
    {
        "name": "CNN_LSTM",
        "model": lambda: CNNLSTM(bidirectional=False),
        "features": mfcc,
        "channel": False
    },
    {
        "name": "CNN_BiLSTM",
        "model": lambda: CNNLSTM(bidirectional=True),
        "features": mfcc,
        "channel": False
    }
]

# =========================================================
# 11. AUTOMATED EXPERIMENT RUNNER
# =========================================================

all_results = []

comparison_path = OUTPUT / "dl_model_comparison.csv"

print("\nStarting automated experiments...")

for experiment in experiments:

    name = experiment["name"]
    features = experiment["features"]

    print("\nPreparing data for:", name)

    train_loader = make_loader(
        features,
        train_idx,
        shuffle=True
    )

    val_loader = make_loader(
        features,
        val_idx
    )

    test_loader = make_loader(
        features,
        test_idx
    )

    model = experiment["model"]()

    try:

        result = train_model(
            name,
            model,
            train_loader,
            val_loader,
            test_loader
        )

        all_results.append(result)

        # Save results after each model
        pd.DataFrame(all_results).sort_values(
            "Macro_F1",
            ascending=False
        ).to_csv(
            comparison_path,
            index=False
        )

        print("\nProgress saved:", comparison_path)

    except Exception as error:

        print("\nERROR in", name)
        print(str(error))

        with open(
            OUTPUT / "training_errors.txt",
            "a",
            encoding="utf-8"
        ) as f:
            f.write(f"\n{name}: {error}\n")

        continue

    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =========================================================
# 12. FINAL COMPARISON
# =========================================================

print("\n" + "=" * 65)
print("DEEP LEARNING EXPERIMENTS COMPLETED")
print("=" * 65)

if all_results:

    final_results = pd.DataFrame(all_results)

    final_results = final_results.sort_values(
        "Macro_F1",
        ascending=False
    )

    final_results.to_csv(
        comparison_path,
        index=False
    )

    print(final_results.to_string(index=False))

    best_model = final_results.iloc[0]

    print("\nBEST MODEL:", best_model["Model"])
    print("TEST ACCURACY:", best_model["Accuracy"])
    print("TEST MACRO F1:", best_model["Macro_F1"])

else:
    print("No models completed successfully.")

print("\nResults:", OUTPUT)
print("Checkpoints:", CHECKPOINTS)
print("Graphs:", GRAPHS)
print("\nALL DONE!")
