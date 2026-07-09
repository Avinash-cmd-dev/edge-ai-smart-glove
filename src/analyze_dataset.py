import pandas as pd

DATASET_PATH = "data/public/gesture_dataset/gesture_dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("="*60)
print("EDGE AI SMART GLOVE")
print("="*60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst Five Rows:")
print(df.head())

print("\n")

print("="*60)
print("Gesture Counts")
print("="*60)

print(df["label"].value_counts())

print("\n")

print("="*60)
print("Missing Values")
print("="*60)

print(df.isnull().sum())

print("\n")

print("="*60)
print("Dataset Information")
print("="*60)

print(df.info())