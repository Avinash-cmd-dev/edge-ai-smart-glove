import pandas as pd

df = pd.read_csv("data/public/gesture_dataset/gesture_dataset.csv")

print("="*60)
print("DATASET INSPECTION")
print("="*60)

print("\nUnique Gestures:")
print(df["label"].unique())

print("\nNumber of Gestures:")
print(df["label"].nunique())

print("\nSamples per Gesture:")
print(df["label"].value_counts())

print("\nUnique sample IDs:")
print(df["sample_id"].nunique())

print("\nRows per sample:")

print(df.groupby("sample_id").size().head(20))