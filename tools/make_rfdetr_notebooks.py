"""Writes the three RF-DETR training notebooks — train_object_detection,
train_segmentation, train_keypoint — from shared cells, so they cannot
drift. Keeps the existing OD notebook's metadata (Colab GPU runtime).

usage: make_rfdetr_notebooks.py <training_notebooks dir> <rfdetr_pkl.py>"""
import json, os, sys

NB_DIR, HELPER = sys.argv[1], sys.argv[2]
meta = json.load(open(os.path.join(NB_DIR, "train_object_detection.ipynb")))["metadata"]
helper = open(HELPER).read().split("\n", 2)[2]          # the helper's body, without its two header comment lines


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(True)}


TASKS = {
    "od": {
        "file": "train_object_detection.ipynb",
        "title": "Custom Object Detection Training — RF-DETR",
        "what": "an object detector (RF-DETR)",
        "export": "**Object Detection**",
        "models": '{"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium,\n          "base": RFDETRBase, "large": RFDETRLarge}',
        "imports": "RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRBase, RFDETRLarge",
        "size": '"small"       # "nano" | "small" | "medium" | "base" | "large"',
        "speed": "On the Jetson Orin Nano Super (FP16 TensorRT), `small` runs a frame in about 15 ms.",
        "resolution": "# None = the size's default (nano 384, small 512, medium 576, base 560,\n                           # large 704); else a multiple of 32 (base: of 56)",
    },
    "seg": {
        "file": "train_segmentation.ipynb",
        "title": "Custom Segmentation Training — RF-DETR Seg",
        "what": "an instance-segmentation model (RF-DETR Seg): each object's box and outline",
        "export": "**Instance Segmentation**",
        "models": '{"nano": RFDETRSegNano, "small": RFDETRSegSmall, "medium": RFDETRSegMedium,\n          "large": RFDETRSegLarge, "xlarge": RFDETRSegXLarge}',
        "imports": "RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge, RFDETRSegXLarge",
        "size": '"small"       # "nano" | "small" | "medium" | "large" | "xlarge"',
        "speed": "On the Jetson Orin Nano Super (FP16 TensorRT), `small` runs a frame in about 25 ms, plus ~2 ms per object for its outline.",
        "resolution": "# None = the size's default; else a multiple of the size's patch grid\n                           # (RF-DETR raises if not)",
    },
    "kp": {
        "file": "train_keypoint.ipynb",
        "title": "Custom Keypoint Training — RF-DETR Keypoints",
        "what": "a keypoint model (RF-DETR Keypoints): every object and its keypoints in one pass",
        "export": "**Keypoint Detection** (COCO format)",
        "models": '{"preview": RFDETRKeypointPreview}',
        "imports": "RFDETRKeypointPreview",
        "size": '"preview"     # RF-DETR\'s keypoint model (its "Preview" release)',
        "speed": "On the Jetson Orin Nano Super (FP16 TensorRT) a frame takes about 55 ms.",
        "resolution": "# None = the model's default (576); else a multiple of 32",
    },
}


def cells(task, t):
    kp_names = ""
    if task == "kp":
        kp_names = '''
keypoint_names = {c["name"]: c["keypoints"] for c in train_coco["categories"] if c.get("keypoints")}
skeleton = {c["name"]: c.get("skeleton", []) for c in train_coco["categories"] if c.get("keypoints")}
for name, kps in keypoint_names.items():
    print(f"  keypoints of '{name}' ({len(kps)}): {kps}")'''
    seg_note = '\n    masks = sum(1 for a in coco.get("annotations", []) if a.get("segmentation"))\n    print(f"           {masks} with masks")' if task == "seg" else ""
    return [
        md(f"""
# {t["title"]}

Train {t["what"]} on your Roboflow dataset (COCO format) and download a `.pkl`
for the Jetson vision server.

The `.pkl` carries the model as **ONNX**. On the Jetson, run `model_prepare`
with it once: that builds its TensorRT engine for the Jetson's GPU (a few
minutes; an engine is tied to the GPU that builds it, so it cannot be built here).

**Important:** Set **Runtime > Change runtime type > GPU** before starting.
"""),
        md(f"## 1. Download Dataset from Roboflow\n\nExport your Roboflow project as {t['export']}."),
        code("""
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY_HERE")
project = rf.workspace("your-workspace").project("your-project")
version = project.version(1)
dataset = version.download("coco")
"""),
        md("## 2. Install RF-DETR"),
        code("""
# Pinned: the version the Jetson's decode (dorna_vision/ai.py) is checked
# against — same ONNX inputs/outputs. Bump both together.
!pip install -q "rfdetr[train,onnx]==1.11.2"
"""),
        md(f"## 3. Training Parameters\n\n{t['speed']}"),
        code(f"""
import os
import json
import pickle
from google.colab import files
from rfdetr import {t["imports"]}

# ============================================================
# EDIT THIS SECTION
# ============================================================
MODEL_SIZE = {t["size"]}
EPOCHS = 100
BATCH_SIZE = 8
GRAD_ACCUM = 2             # effective batch = BATCH_SIZE x GRAD_ACCUM (16 recommended)
LR = 1e-4
RESOLUTION = None          {t["resolution"]}

MODELS = {t["models"]}
Model = MODELS[MODEL_SIZE]

project_name = project.name.lower().replace(" ", "_")

print(f"Model size:  {{MODEL_SIZE}}")
print(f"Epochs:      {{EPOCHS}}")
print(f"Batch size:  {{BATCH_SIZE}} x {{GRAD_ACCUM}}")
print(f"Resolution:  {{RESOLUTION or 'model default'}}")
"""),
        md("""
## 4. Verify COCO Dataset Layout

Roboflow's COCO export is RF-DETR's native format — no conversion needed.
This cell sanity-checks the paths and reports what each split holds.
"""),
        code(f"""
# Roboflow's COCO export layout:
#   dataset.location/
#     train/_annotations.coco.json + images
#     valid/_annotations.coco.json + images
#     test/_annotations.coco.json  + images

DATA_ROOT = dataset.location

for split in ("train", "valid", "test"):
    ann_path = f"{{DATA_ROOT}}/{{split}}/_annotations.coco.json"
    if not os.path.exists(ann_path):
        print(f"  {{split}}: MISSING  ({{ann_path}})")
        continue
    with open(ann_path) as f:
        coco = json.load(f)
    print(f"  {{split}}: {{len(coco.get('images', []))}} images, {{len(coco.get('annotations', []))}} annotations"){seg_note}

with open(f"{{DATA_ROOT}}/train/_annotations.coco.json") as f:
    train_coco = json.load(f){kp_names}
"""),
        md("## 5. Train"),
        code("""
OUTPUT_DIR = "/content/output"

model = Model()
model.train(
    dataset_dir=DATA_ROOT,
    output_dir=OUTPUT_DIR,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    grad_accum_steps=GRAD_ACCUM,
    lr=LR,
    **({"resolution": RESOLUTION} if RESOLUTION else {}),
)
"""),
        md("""
## 6. Export ONNX

The best checkpoint (by validation mAP) is exported. The check runs the ONNX
file on a validation image and compares its best box with RF-DETR's own
prediction — a bad export shows here, not on the bench.
"""),
        code("""
import cv2
import numpy as np
import onnxruntime as ort

best = Model(pretrain_weights=f"{OUTPUT_DIR}/checkpoint_best_total.pth")
onnx_path = best.export(output_dir="/content/export", verbose=False)

S = int(best.model_config.resolution)
val_dir = f"{DATA_ROOT}/valid"
sample = sorted(f for f in os.listdir(val_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))[0]
img = cv2.imread(f"{val_dir}/{sample}")
H, W = img.shape[:2]

# RF-DETR's predict() preprocessing: RGB, resize to S x S, ImageNet mean/std
rgb = cv2.resize(img, (S, S), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].astype(np.float32) / 255
x = ((rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32)
session = ort.InferenceSession(onnx_path)
out = dict(zip([o.name for o in session.get_outputs()], session.run(None, {"input": x})))
q = (1 / (1 + np.exp(-out["labels"][0]))).max(1).argmax()
cx, cy, w, h = out["dets"][0][q]
onnx_box = np.array([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])

pred = best.predict(img[:, :, ::-1].copy(), threshold=0.0)
boxes = pred.xyxy if hasattr(pred, "xyxy") else pred.data["xyxy"]
scores = pred.confidence if hasattr(pred, "xyxy") else pred.detection_confidence
print("ONNX best box:   ", np.round(onnx_box, 1))
print("RF-DETR best box:", np.round(boxes[int(np.argmax(scores))], 1))
"""),
        md("## 7. Save & Download Pickle"),
        code(helper.strip("\n")),
        code(f"""
try:
    colors = project.colors or {{}}
except Exception:
    colors = {{}}

model_dict = rfdetr_model_dict(best, onnx_path, "{task}", MODEL_SIZE, colors{", keypoint_names=keypoint_names, skeleton=skeleton" if task == "kp" else ""})

pickle_path = f"/content/{{project_name}}.pkl"
with open(pickle_path, "wb") as f:
    pickle.dump(model_dict, f)

print(f"Saved: {{pickle_path}}")
print(f"Total size: {{os.path.getsize(pickle_path) / 1024 / 1024:.1f}} MB")
print(f"Classes: {{[c for c in model_dict['cls'] if c is not None]}}")
print(f"Image size: {{model_dict['meta']['image_size']}}")

files.download(pickle_path)
"""),
    ]


for task, t in TASKS.items():
    nb = {"cells": cells(task, t), "metadata": meta, "nbformat": 4, "nbformat_minor": 0}
    path = os.path.join(NB_DIR, t["file"])
    json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
    open(path, "a").write("\n")
    print("wrote", path, len(nb["cells"]), "cells")
