
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE = Path(__file__).resolve().parent.parent

OUTPUT = BASE / "outputs"
MODELS = BASE / "models"

OUTPUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

FEATURE_FILE = OUTPUT / "X_features.npy"
LABEL_FILE = OUTPUT / "labels.csv"
COMPARISON_FILE = OUTPUT / "model_comparison.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 4

# ==================================================
# 2. LOAD AND VALIDATE DATASET
# ==================================================

print("\n" + "=" * 65)
print("HUMAN EMOTION DETECTION - ML MODEL COMPARISON")
print("=" * 65)

print("\nLoading dataset...")

if not FEATURE_FILE.exists():
    raise FileNotFoundError(f"Feature file missing: {FEATURE_FILE}")

if not LABEL_FILE.exists():
    raise FileNotFoundError(f"Label file missing: {LABEL_FILE}")

df = pd.read_csv(LABEL_FILE)
X = np.load(FEATURE_FILE).astype(np.float32)

required_columns = {"filename", "emotion"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"labels.csv must contain columns: {required_columns}"
    )

if len(X) != len(df):
    raise ValueError(
        f"Feature-label mismatch: {len(X)} features, {len(df)} labels"
    )

if not np.isfinite(X).all():
    raise ValueError("Feature matrix contains NaN or infinite values.")

y = df["emotion"].astype(str).to_numpy()

# Extract actor IDs from RAVDESS filenames.
# Example: 03-01-05-01-02-01-12.wav -> Actor 12

groups = np.array([
    Path(str(filename)).stem.split("-")[-1]
    for filename in df["filename"]
])

if len(np.unique(groups)) < CV_FOLDS:
    raise ValueError("Not enough unique actors for GroupKFold.")

print("Feature shape:", X.shape)
print("Total recordings:", len(X))
print("Emotion classes:", np.unique(y))
print("Total actors:", len(np.unique(groups)))

print("\nEmotion distribution:")
print(pd.Series(y).value_counts())

# ==================================================
# 3. ACTOR-INDEPENDENT TRAIN / TEST SPLIT
# ==================================================

print("\nCreating actor-independent train/test split...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_train = X[train_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

groups_train = groups[train_idx]
groups_test = groups[test_idx]

# Verify there is no actor overlap.
overlap = set(groups_train) & set(groups_test)

if overlap:
    raise RuntimeError(f"Actor leakage detected: {overlap}")

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("Training actors:", len(np.unique(groups_train)))
print("Testing actors:", len(np.unique(groups_test)))

print("Actor overlap:", len(overlap))

# ==================================================
# 4. DEFINE MACHINE LEARNING MODELS
# ==================================================

experiments = {

    "SVM": (
        Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ]),
        {
            "model__C": [10, 30],
            "model__gamma": [0.01, 0.001],
            "model__kernel": ["rbf"]
        }
    ),

    "RandomForest": (
        RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        {
            "n_estimators": [200, 400],
            "max_depth": [None, 20],
            "min_samples_leaf": [1, 2]
        }
    ),

    "ExtraTrees": (
        ExtraTreesClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        {
            "n_estimators": [200, 400],
            "max_features": ["sqrt", 0.7],
            "min_samples_leaf": [1, 2]
        }
    ),

    "KNN": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ]),
        {
            "model__n_neighbors": [5, 7, 9],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2]
        }
    )
}

# ==================================================
# 5. GROUP CROSS-VALIDATION
# ==================================================

cv = GroupKFold(n_splits=CV_FOLDS)

results = []

# ==================================================
# 6. TRAIN AND EVALUATE
# ==================================================

for name, (estimator, parameters) in experiments.items():

    print("\n" + "=" * 65)
    print(f"TRAINING MODEL: {name}")
    print("=" * 65)

    search = GridSearchCV(
        estimator=estimator,
        param_grid=parameters,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        verbose=1,
        refit=True,
        error_score="raise"
    )

    # Actor-independent cross-validation.
    search.fit(
        X_train,
        y_train,
        groups=groups_train
    )

    best_model = search.best_estimator_

    # Evaluate only on the held-out test actors.
    predictions = best_model.predict(X_test)

    # ==================================================
    # METRICS
    # ==================================================

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),
        "Macro_F1": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),
        "Best_CV_Macro_F1": search.best_score_
    }

    results.append(metrics)

    # ==================================================
    # SAVE MODEL IMMEDIATELY
    # ==================================================

    model_path = MODELS / f"{name.lower()}_model.joblib"

    joblib.dump(best_model, model_path)

    print("\nModel saved:", model_path)

    print("\nBest hyperparameters:")
    print(json.dumps(search.best_params_, indent=4))

    print("\nTest metrics:")
    print(json.dumps(metrics, indent=4))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # ==================================================
    # SAVE PROGRESS AFTER EACH MODEL
    # ==================================================

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Macro_F1",
        ascending=False
    )

    results_df.to_csv(
        COMPARISON_FILE,
        index=False
    )

    print("\nProgress saved:", COMPARISON_FILE)

# ==================================================
# 7. FINAL MODEL COMPARISON
# ==================================================

print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro_F1",
    ascending=False
).reset_index(drop=True)

print(results_df.to_string(index=False))

results_df.to_csv(
    COMPARISON_FILE,
    index=False
)

# ==================================================
# 8. BEST MODEL
# ==================================================

best_result = results_df.iloc[0]

print("\n" + "=" * 65)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 65)

print("Best model:", best_result["Model"])
print(f"Accuracy: {best_result['Accuracy']:.4f}")
print(f"Macro F1: {best_result['Macro_F1']:.4f}")
print(f"Precision: {best_result['Precision']:.4f}")
print(f"Recall: {best_result['Recall']:.4f}")

print("\nSaved files:")
print("Models:", MODELS)
print("Comparison:", COMPARISON_FILE)

print("\nAll experiments completed.")
