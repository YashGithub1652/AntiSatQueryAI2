import sys, numpy as np, glob, os
from PIL import Image

sys.path.insert(0, ".")
from satquery.backend.models.change_engine import ChangeDetectionEngine

root = "external/ChangeFormer/samples_LEVIR"
files = sorted(glob.glob(root + "/label/*.png"))
engine = ChangeDetectionEngine()

TP = FP = FN = TN = 0

for f in files:
    name = os.path.basename(f)

    gt = (np.asarray(Image.open(f).convert("L")) > 0).astype(np.uint8)

    a = np.asarray(
        Image.open(root + "/A/" + name).convert("RGB"),
        dtype=np.float32
    ) / 255.0

    b = np.asarray(
        Image.open(root + "/B/" + name).convert("RGB"),
        dtype=np.float32
    ) / 255.0

    pred = engine.run(a, b)["_change_map_raw"].astype(np.uint8)

    TP += int(((pred == 1) & (gt == 1)).sum())
    FP += int(((pred == 1) & (gt == 0)).sum())
    FN += int(((pred == 0) & (gt == 1)).sum())
    TN += int(((pred == 0) & (gt == 0)).sum())

iou = TP / (TP + FP + FN) if TP + FP + FN else 0
precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

print("SAMPLES:", len(files))
print("TP:", TP)
print("FP:", FP)
print("FN:", FN)
print("TN:", TN)
print("DATASET_IoU:", round(iou, 6))
print("DATASET_PRECISION:", round(precision, 6))
print("DATASET_RECALL:", round(recall, 6))
print("DATASET_F1:", round(f1, 6))
