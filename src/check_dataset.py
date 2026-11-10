
from pathlib import Path

DATASET = Path(r"G:\My Drive\Human_Emotion_Detection\dataset")

actor_folders = sorted(DATASET.glob("Actor_*"))

audio_files = list(DATASET.rglob("*.wav"))

print("=" * 45)
print("RAVDESS DATASET VERIFICATION")
print("=" * 45)

print("Actor folders:", len(actor_folders))
print("Audio recordings:", len(audio_files))

for actor in actor_folders[:5]:
    count = len(list(actor.glob("*.wav")))
    print(f"{actor.name}: {count} recordings")

if len(actor_folders) == 24 and len(audio_files) == 1440:
    print("\nDATASET VERIFIED SUCCESSFULLY!")
else:
    print("\nCheck the extracted folder structure.")
