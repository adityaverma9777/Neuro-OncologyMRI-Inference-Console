import os

import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = "results.csv"
OUT_DIR = "plots"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

print("Columns found:", df.columns.tolist())


def get_col(name):
    return name if name in df.columns else None

plt.figure()
if get_col("metrics/mAP50(B)"):
    plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")
if get_col("metrics/mAP50-95(B)"):
    plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Model Accuracy (mAP)")
plt.legend()
plt.grid()
plt.savefig(os.path.join(OUT_DIR, "map.png"))
plt.close()
plt.figure()
if get_col("train/box_loss"):
    plt.plot(df["epoch"], df["train/box_loss"], label="Box Loss")
if get_col("train/cls_loss"):
    plt.plot(df["epoch"], df["train/cls_loss"], label="Class Loss")
if get_col("train/dfl_loss"):
    plt.plot(df["epoch"], df["train/dfl_loss"], label="DFL Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid()
plt.savefig(os.path.join(OUT_DIR, "loss.png"))
plt.close()
plt.figure()
if get_col("metrics/precision(B)"):
    plt.plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
if get_col("metrics/recall(B)"):
    plt.plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Precision vs Recall")
plt.legend()
plt.grid()
plt.savefig(os.path.join(OUT_DIR, "precision_recall.png"))
plt.close()

print(f"Graphs saved in folder: {OUT_DIR}")
