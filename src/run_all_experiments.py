"""
Automated RAVDESS Speech Emotion Recognition experiments.

Models:
- CNN2D_Mel
- CNN_BiGRU_Attention
- CNN_Transformer
- Wav2Vec2-base (frozen encoder + trained classifier)
- HuBERT-base (frozen encoder + trained classifier)
- WavLM-base-plus (frozen encoder + trained classifier)
- AST (Audio Spectrogram Transformer, frozen encoder + trained classifier)

All models use the SAME actor-independent train/validation/test split.
This script is an experimental baseline pipeline, not a guarantee of high accuracy.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
OUT_DIR = ROOT / "outputs" / "automated"
MODEL_DIR = ROOT / "models" / "automated"
CACHE_DIR = OUT_DIR / "cache"

SAMPLE_RATE = 16000
DURATION = 4.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 35
PATIENCE = 8
SEED = 42
NUM_WORKERS = 0  # safest default on Windows

EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

PRETRAINED = {
    "Wav2Vec2": ("facebook/wav2vec2-base", "wav2vec2"),
    "HuBERT": ("facebook/hubert-base-ls960", "hubert"),
    "WavLM": ("microsoft/wavlm-base-plus", "wavlm"),
    "AST": ("MIT/ast-finetuned-audioset-10-10-0.4593", "ast"),
}


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def discover_dataset():
    rows = []
    for wav in sorted(DATA_DIR.glob("Actor_*/*.wav")):
        parts = wav.stem.split("-")
        if len(parts) != 7:
            continue
        emotion_code = parts[2]
        actor = parts[6]
        if emotion_code not in EMOTION_MAP:
            continue
        rows.append({
            "path": str(wav),
            "filename": wav.name,
            "emotion": EMOTION_MAP[emotion_code],
            "actor": actor
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise FileNotFoundError(
            f"No RAVDESS WAV files found. Expected: {DATA_DIR}\\Actor_01\\*.wav"
        )
    if len(df) != 1440:
        print(f"WARNING: Found {len(df)} WAV files; standard RAVDESS speech set has 1440.")
    if df["actor"].nunique() < 5:
        raise RuntimeError("Too few unique actors for a meaningful grouped split.")
    return df


def make_splits(df):
    # 20% actors held out for test; 20% of remaining actors for validation.
    idx = np.arange(len(df))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    trainval_idx, test_idx = next(gss1.split(idx, groups=df["actor"].values))
    trainval = df.iloc[trainval_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED + 1)
    tr_rel, va_rel = next(gss2.split(trainval_idx, groups=trainval["actor"].values))
    train_idx = trainval_idx[tr_rel]
    val_idx = trainval_idx[va_rel]

    sets = {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }
    actor_sets = {k: set(v.actor) for k, v in sets.items()}
    assert actor_sets["train"].isdisjoint(actor_sets["val"])
    assert actor_sets["train"].isdisjoint(actor_sets["test"])
    assert actor_sets["val"].isdisjoint(actor_sets["test"])
    return sets


def load_audio(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(y) > N_SAMPLES:
        y = y[:N_SAMPLES]
    elif len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    return y.astype(np.float32)


def extract_mel(df, cache_path):
    if cache_path.exists():
        cached = np.load(cache_path)
        if len(cached["features"]) == len(df):
            return cached["features"]
    feats = []
    for i, row in enumerate(df.itertuples(index=False), 1):
        y = load_audio(row.path)
        mel = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmin=20, fmax=7600, power=2.0
        )
        logmel = librosa.power_to_db(mel, ref=np.max)
        # Per-utterance normalization for stable CNN training.
        logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
        feats.append(logmel.astype(np.float32))
        if i % 200 == 0 or i == len(df):
            print(f"  Mel features: {i}/{len(df)}")
    arr = np.stack(feats)
    np.savez_compressed(cache_path, features=arr)
    return arr


class MelDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]


class CNN2D(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.20),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(128, n_classes)
        )
    def forward(self, x): return self.net(x.unsqueeze(1))


class CNNBiGRUAttention(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(N_MELS, 128, 5, padding=2), nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(2), nn.Dropout(0.15),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU()
        )
        self.gru = nn.GRU(128, 96, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(nn.Linear(192, 96), nn.Tanh(), nn.Linear(96, 1))
        self.head = nn.Sequential(nn.Linear(192, 128), nn.ReLU(), nn.Dropout(0.35),
                                  nn.Linear(128, n_classes))
    def forward(self, x):
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        z, _ = self.gru(z)
        weights = torch.softmax(self.attn(z).squeeze(-1), dim=1)
        pooled = torch.sum(z * weights.unsqueeze(-1), dim=1)
        return self.head(pooled)


class CNNTransformer(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(N_MELS, 128, 5, padding=2), nn.GELU(),
            nn.MaxPool1d(2), nn.Conv1d(128, 128, 3, padding=1), nn.GELU()
        )
        layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.2,
            batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.Linear(128, 128), nn.GELU(),
                                  nn.Dropout(0.35), nn.Linear(128, n_classes))
    def forward(self, x):
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        z = self.transformer(z)
        return self.head(z.mean(dim=1))


def train_neural_model(name, model, x, y, splits, label_encoder, device):
    train_idx, val_idx, test_idx = splits
    # CNN models receive [N, time, mel].
    x_t = np.transpose(x, (0, 2, 1)).astype(np.float32)
    train_ds = MelDataset(x_t[train_idx], y[train_idx])
    val_ds = MelDataset(x_t[val_idx], y[val_idx])
    test_ds = MelDataset(x_t[test_idx], y[test_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS)

    classes = np.unique(y[train_idx])
    weights = compute_class_weight("balanced", classes=classes, y=y[train_idx])
    class_weights = torch.ones(len(label_encoder.classes_), dtype=torch.float32)
    class_weights[classes] = torch.tensor(weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_f1, best_state, wait = -1.0, None, 0
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        val_true, val_pred = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb.to(device)).argmax(1).cpu().numpy()
                val_pred.extend(pred); val_true.extend(yb.numpy())
        vf1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        print(f"  {name} epoch {epoch:02d}: validation macro-F1={vf1:.4f}")
        if vf1 > best_f1:
            best_f1, wait = vf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                print("  Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "classes": label_encoder.classes_.tolist(),
                "model_name": name}, MODEL_DIR / f"{name}.pt")

    true, pred = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            pred.extend(model(xb.to(device)).argmax(1).cpu().numpy())
            true.extend(yb.numpy())
    return evaluate_and_save(name, true, pred, label_encoder, time.time() - start)


def extract_transformer_embeddings(df, model_key, model_id, cache_path, device):
    if cache_path.exists():
        z = np.load(cache_path)
        if len(z["features"]) == len(df):
            print(f"  Using cached {model_key} embeddings.")
            return z["features"]

    from transformers import AutoFeatureExtractor, AutoModel
    print(f"  Loading {model_id} ...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    encoder = AutoModel.from_pretrained(model_id)
    encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad = False

    embeddings = []
    is_ast = model_key == "AST"
    target_sr = 16000
    for i, row in enumerate(df.itertuples(index=False), 1):
        audio = load_audio(row.path)
        inputs = feature_extractor(
            audio, sampling_rate=target_sr, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            out = encoder(**inputs)
            hidden = out.last_hidden_state
            # AST uses a sequence of spectrogram patch embeddings; other models
            # use frame-level contextual embeddings.
            pooled = hidden.mean(dim=1).squeeze(0)
        embeddings.append(pooled.float().cpu().numpy())
        if i % 100 == 0 or i == len(df):
            print(f"  {model_key} embeddings: {i}/{len(df)}")

    arr = np.stack(embeddings).astype(np.float32)
    np.savez_compressed(cache_path, features=arr)
    del encoder, feature_extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return arr


class EmbeddingDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]


class MLPHead(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(0.40), nn.Linear(256, 128), nn.GELU(),
            nn.Dropout(0.30), nn.Linear(128, n_classes)
        )
    def forward(self, x): return self.net(x)


def train_embedding_head(name, features, y, split_indices, label_encoder, device):
    tr, va, te = split_indices
    # Fit normalization on training data only: no leakage.
    scaler = StandardScaler()
    xtr = scaler.fit_transform(features[tr]).astype(np.float32)
    xva = scaler.transform(features[va]).astype(np.float32)
    xte = scaler.transform(features[te]).astype(np.float32)

    train_ds = EmbeddingDataset(xtr, y[tr])
    val_ds = EmbeddingDataset(xva, y[va])
    test_ds = EmbeddingDataset(xte, y[te])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    classes = np.unique(y[tr])
    weights = compute_class_weight("balanced", classes=classes, y=y[tr])
    cw = torch.ones(len(label_encoder.classes_), dtype=torch.float32)
    cw[classes] = torch.tensor(weights, dtype=torch.float32)
    model = MLPHead(features.shape[1], len(label_encoder.classes_)).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    best_f1, best_state, wait = -1.0, None, 0
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        vt, vp = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                vp.extend(model(xb.to(device)).argmax(1).cpu().numpy())
                vt.extend(yb.numpy())
        vf1 = f1_score(vt, vp, average="macro", zero_division=0)
        print(f"  {name} epoch {epoch:02d}: validation macro-F1={vf1:.4f}")
        if vf1 > best_f1:
            best_f1, wait = vf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                print("  Early stopping.")
                break

    model.load_state_dict(best_state)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "classes": label_encoder.classes_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "model_name": name
    }, MODEL_DIR / f"{name}.pt")

    tt, tp = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            tp.extend(model(xb.to(device)).argmax(1).cpu().numpy())
            tt.extend(yb.numpy())
    return evaluate_and_save(name, tt, tp, label_encoder, time.time() - start)


def evaluate_and_save(name, y_true, y_pred, encoder, elapsed):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = np.arange(len(encoder.classes_))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=encoder.classes_,
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(report).transpose().to_csv(OUT_DIR / f"{name}_classification_report.csv")
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title(f"{name} — Test confusion matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}_confusion_matrix.png", dpi=160)
    plt.close()
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "training_seconds": elapsed
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pretrained", action="store_true",
                        help="Run only log-Mel CNN models.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Optional model names: CNN2D CNN_BiGRU CNN_Transformer Wav2Vec2 HuBERT WavLM AST")
    parser.add_argument("--force-reextract", action="store_true")
    args = parser.parse_args()

    seed_everything()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("AUTOMATED SPEECH EMOTION RECOGNITION EXPERIMENTS")
    print("=" * 70)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU memory:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")

    df = discover_dataset()
    df.to_csv(OUT_DIR / "dataset_manifest.csv", index=False)
    print(f"Dataset: {len(df)} recordings | {df.actor.nunique()} actors | {df.emotion.nunique()} emotions")
    splits = make_splits(df)
    split_indices = []
    # Store indices by filename so all feature families share the exact same partition.
    for part in ("train", "val", "test"):
        split_indices.append(df.index[df.filename.isin(splits[part].filename)].to_numpy())
    tr, va, te = split_indices
    print(f"Split sizes: train={len(tr)}, validation={len(va)}, test={len(te)}")
    print("Actor groups:", {k: sorted(v.actor.unique().tolist()) for k, v in splits.items()})
    with open(OUT_DIR / "split_actors.json", "w", encoding="utf-8") as f:
        json.dump({k: sorted(v.actor.unique().tolist()) for k, v in splits.items()}, f, indent=2)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df.emotion)
    results = []
    requested = set(args.only) if args.only else None

    def wanted(name):
        return requested is None or name in requested

    mel_path = CACHE_DIR / "logmel_features.npz"
    if args.force_reextract and mel_path.exists(): mel_path.unlink()
    mel = extract_mel(df, mel_path)

    neural_specs = [
        ("CNN2D", CNN2D(len(encoder.classes_))),
        ("CNN_BiGRU", CNNBiGRUAttention(len(encoder.classes_))),
        ("CNN_Transformer", CNNTransformer(len(encoder.classes_))),
    ]
    for name, model in neural_specs:
        if not wanted(name): continue
        print("\n" + "=" * 60 + f"\nTRAINING {name}\n" + "=" * 60)
        try:
            results.append(train_neural_model(name, model, mel, y, split_indices, encoder, device))
        except Exception as exc:
            print(f"FAILED {name}: {exc}")
            traceback.print_exc()
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    if not args.skip_pretrained:
        for name, (model_id, kind) in PRETRAINED.items():
            if not wanted(name): continue
            print("\n" + "=" * 60 + f"\nEXPERIMENT: {name}\n" + "=" * 60)
            cache_path = CACHE_DIR / f"{name.lower()}_embeddings.npz"
            try:
                features = extract_transformer_embeddings(df, name, model_id, cache_path, device)
                results.append(train_embedding_head(name, features, y, split_indices, encoder, device))
                del features
            except Exception as exc:
                print(f"FAILED {name}: {exc}")
                traceback.print_exc()
            finally:
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

    if not results:
        raise RuntimeError("No experiment completed successfully. See errors above.")
    result_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    result_df.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    result_df.to_json(OUT_DIR / "model_comparison.json", orient="records", indent=2)

    plt.figure(figsize=(11, 6))
    plot_df = result_df.sort_values("macro_f1", ascending=True)
    plt.barh(plot_df["model"], plot_df["macro_f1"])
    plt.xlabel("Test Macro F1")
    plt.title("Speaker-independent model comparison — Macro F1")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_comparison_macro_f1.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    plot_df = result_df.sort_values("accuracy", ascending=True)
    plt.barh(plot_df["model"], plot_df["accuracy"])
    plt.xlabel("Test accuracy")
    plt.title("Speaker-independent model comparison — Accuracy")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_comparison_accuracy.png", dpi=180)
    plt.close()

    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY (sorted by test macro F1)")
    print("=" * 70)
    print(result_df.to_string(index=False, formatters={
        "accuracy": "{:.4f}".format, "macro_f1": "{:.4f}".format,
        "precision_macro": "{:.4f}".format, "recall_macro": "{:.4f}".format
    }))
    print("\nBest by macro F1:", result_df.iloc[0]["model"])
    print("Results folder:", OUT_DIR)
    print("Checkpoints folder:", MODEL_DIR)
    print("IMPORTANT: Use this held-out test set only for final evaluation, not model tuning.")


if __name__ == "__main__":
    main()
