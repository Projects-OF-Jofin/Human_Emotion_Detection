# Human Emotion Detection from Speech

A Speech Emotion Recognition (SER) project that uses machine learning, deep learning, and pretrained audio transformer models to classify human emotions from speech recordings.

## Overview

This project explores how emotional information can be identified from speech signals using acoustic features and deep learning architectures.

The system experiments with traditional machine learning classifiers, custom neural networks, and pretrained audio representation models.

## Dataset

- **Dataset:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Audio recordings:** 1,440
- **Actors:** 24
- **Emotion classes:** 8

Emotion classes:
- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

The dataset is not included in this repository. Obtain it from its official distribution source and review its usage terms.

## Models Implemented

### Traditional Machine Learning
- Support Vector Machine (SVM)
- Random Forest
- Extra Trees
- K-Nearest Neighbors (KNN)

### Custom Deep Learning
- CNN with 2D Mel Spectrograms
- CNN with 1D MFCC features
- CNN-LSTM
- CNN-BiLSTM
- CNN-BiGRU with attention
- CNN-Transformer

### Pretrained Audio Models
- Wav2Vec2
- HuBERT
- WavLM
- Audio Spectrogram Transformer (AST)

The pretrained models in the automated experiments use frozen feature encoders with trained classification heads; they are not full end-to-end fine-tuned models.

## Best Automated Experiment Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| AST | 56.67% | 58.55% |
| WavLM | 55.67% | 56.44% |
| HuBERT | 55.33% | 56.08% |
| CNN-BiGRU | 49.00% | 48.20% |
| Wav2Vec2 | 46.00% | 46.67% |
| CNN-Transformer | 43.33% | 42.33% |
| CNN2D | 33.00% | 30.05% |

AST achieved the highest macro F1 score among the automated experiments.

These results are specific to the experiment's evaluation setup and should not be interpreted as a guarantee of performance on unseen real-world speech.

## Evaluation

The project records:
- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score
- Classification reports
- Confusion matrices
- Training curves and model comparisons

See `final_results/` for the curated experiment results and visualizations.

## Project Structure

```text
Human_Emotion_Detection/
├── src/
│   ├── check_dataset.py
│   ├── extract_features.py
│   ├── train_compare_ml.py
│   ├── train_all.py
│   ├── train_wav2vec2.py
│   └── run_all_experiments.py
├── final_results/
│   ├── comparison/
│   ├── confusion_matrices/
│   ├── classification_reports/
│   └── visualizations/
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd Human_Emotion_Detection
```

2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

4. Install the dependencies:

```bash
pip install -r requirements.txt
```

5. Install a compatible PyTorch build separately, following the official PyTorch installation instructions for your hardware.

6. Download and arrange the RAVDESS dataset locally. Update the dataset path in the relevant scripts if necessary.

## Running Experiments

Check the dataset:

```bash
python src/check_dataset.py
```

Extract acoustic features:

```bash
python src/extract_features.py
```

Run traditional machine learning experiments:

```bash
python src/train_compare_ml.py
```

Run custom deep learning experiments:

```bash
python src/train_all.py
```

Run pretrained audio model experiments:

```bash
python src/run_all_experiments.py
```

## Hardware

Experiments were conducted using an NVIDIA GeForce RTX 3050 Laptop GPU with 6 GB VRAM and CUDA-enabled PyTorch.

## Limitations

- RAVDESS is a controlled acted-emotion dataset and may not represent all real-world speech conditions.
- Emotion recognition is inherently ambiguous; vocal expression does not always reflect a person's actual emotional state.
- Performance may vary across speakers, recording environments, accents and datasets.
- Further validation on independent real-world datasets is needed.

## License

Add a suitable license after confirming the licensing terms of your own code and the dataset's separate usage conditions.