"""Writes training_notebooks/jetson/train_classification.ipynb (a linear head on
the Jetson's built-in backbone, C-RADIOv3-B) — keeps the existing notebook's
metadata (Colab GPU runtime)."""
import json, sys

path = sys.argv[1]
old = json.load(open(path))

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(True)}

cells = [
md("""
# Image Classification Training — C-RADIOv3-B

Train an image classifier on your Roboflow dataset (folder format) and download a
`.pkl` for the Jetson vision server.

**NVIDIA C-RADIOv3-B**, frozen, with a linear head on its image summary. Nothing to
tune: no augmentation, no epochs, no learning rate. The head's one setting (its
regularization) is picked automatically by cross-validation on the training
images. Training takes a minute.

The backbone is built into the Jetson (its upgrade downloads it and builds its
engine), so the `.pkl` carries only the head — under 100 KB — and is ready the moment
it reaches the unit.

C-RADIOv3-B is released under the **NVIDIA Open Model License** (commercial use
allowed).

**Important:** Set **Runtime > Change runtime type > GPU** before starting.
"""),
md("## 1. Download Dataset from Roboflow"),
code("""
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY_HERE")
project = rf.workspace("your-workspace").project("your-project")
version = project.version(1)
dataset = version.download("folder")
"""),
md("## 2. Install Dependencies"),
code("""
# open_clip: the backbone's Hugging Face code imports it (an adaptor it does not use here)
!pip install -q transformers timm open_clip_torch
"""),
md("""
## 3. Image Features

Every image goes through exactly the Jetson's preprocessing (`dorna_vision/ai/trt.py`
`CLS`): centred letterbox to 224 x 224 with grey padding, RGB, 0..1 — the backbone
normalizes it itself. The backbone is the exact release the Jetson runs: the same
repository and revision; a classifier trained on another is refused there.
"""),
code("""
import os
import cv2
import numpy as np
import torch
from transformers import AutoModel

# Chosen by benchmark (capped crops, sessions two hours apart, 2026-11-17): C-RADIOv3-B
# made 30-45 % fewer mistakes than DINOv3 ViT-S/16 and beat C-RADIOv4-SO400M with 20
# bad images per kind; 7 ms a check on the Jetson.
BACKBONE = {"name": "c-radio_v3-b", "repo": "nvidia/C-RADIOv3-B",
            "revision": "44653a0482cf460bb4f12595fc3cc3dfecc403d1"}
IMAGE_SIZE = (224, 224)   # (W, H)


def preprocess(img_bgr):
    # the Jetson's _letterbox_centered, verbatim; then RGB 0..1
    h, w = img_bgr.shape[:2]
    tw, th = IMAGE_SIZE
    r = min(tw / w, th / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((th, tw, 3), 128, dtype=np.uint8)
    pw, ph = (tw - nw) // 2, (th - nh) // 2
    padded[ph:ph + nh, pw:pw + nw] = resized
    return (cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).transpose(2, 0, 1)


DATA_ROOT = dataset.location
classes = sorted(d for d in os.listdir(f"{DATA_ROOT}/train") if os.path.isdir(f"{DATA_ROOT}/train/{d}"))
project_name = project.name.lower().replace(" ", "_")


def load_split(split):
    paths, labels = [], []
    for ci, c in enumerate(classes):
        folder = f"{DATA_ROOT}/{split}/{c}"
        if os.path.isdir(folder):
            for f in sorted(os.listdir(folder)):
                paths.append(f"{folder}/{f}"); labels.append(ci)
    return paths, np.array(labels)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = AutoModel.from_pretrained(BACKBONE["repo"], revision=BACKBONE["revision"],
                                     trust_remote_code=True).to(device).eval()


@torch.no_grad()
def features(paths):
    out = []
    for i in range(0, len(paths), 32):
        batch = np.stack([preprocess(cv2.imread(p)) for p in paths[i:i + 32]])
        out.append(backbone(torch.from_numpy(batch).to(device))[0].float().cpu().numpy())   # [0]: the summary
    return np.concatenate(out)


train_paths, y_train = load_split("train")
val_paths, y_val = load_split("valid")
X_train, X_val = features(train_paths), features(val_paths)

print(f"Classes ({len(classes)}): {classes}")
print(f"Train: {len(train_paths)} images (per class: {np.bincount(y_train, minlength=len(classes)).tolist()})")
print(f"Valid: {len(val_paths)} images")
print(f"Features: {X_train.shape[1]} per image")
"""),
md("""
## 4. Train the Linear Head

Logistic regression on the features, every class weighted equally (a few fail images
next to many good ones count as much); its regularization is picked by 5-fold
cross-validation on the training images only — the validation images stay unseen.
"""),
code("""
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# class_weight "balanced": every class counts equally however many images it has — a few
# fail images next to many good ones otherwise pull the head toward "good"
head = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, cv=5, scoring="accuracy", max_iter=5000,
                                                            class_weight="balanced"))
head.fit(X_train, y_train)

probs = head.predict_proba(X_val)
pred = probs.argmax(1)
labels = y_val
print(f"Train acc: {100 * head.score(X_train, y_train):.2f}%")
print(f"Valid acc: {100 * (pred == labels).mean():.2f}%  ({int((pred != labels).sum())} wrong of {len(labels)})")
"""),
md("""
## Confusion matrix and the decision threshold

Accuracy hides the number that matters for a sorter: how many **fails were called pass**. The first
matrix row is the fails — caught, then missed. The table then shows what each threshold on the
`pass` probability costs on both sides (fails missed vs good parts rejected); the project's `Sort`
uses that threshold instead of the plain argmax when missed fails must be rarer than the 0.5 cut gives.
"""),
code("""
from sklearn.metrics import confusion_matrix, classification_report

print("rows = true", classes, "| cols = predicted", classes)
print(confusion_matrix(labels, pred))
print(classification_report(labels, pred, target_names=classes, digits=3))

# the threshold — only meaningful for a pass/fail problem: call it 'pass' only when the pass
# probability is at least t, else 'fail'. Pick the row whose missed fails you can live with.
if "pass" in classes and "fail" in classes:
    i_pass, i_fail = classes.index("pass"), classes.index("fail")
    print("t      missed fails   good parts rejected")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        p = np.where(probs[:, i_pass] >= t, i_pass, i_fail)
        cm = confusion_matrix(labels, p, labels=[i_fail, i_pass])
        print(f"{t:<6} {cm[0,1]:>8}        {cm[1,0]:>8}")
"""),
md("""
## The misclassified validation images

One gallery per true class: its images the model put in another class, titled with the file,
the predicted class and its probability. Works for any set of classes. Read them before
anything else: a wrong label, a defect the crop cannot show, a failure mode with two examples
— each asks for a different fix.
"""),
code("""
import matplotlib.pyplot as plt

misclassified = {c: [i for i in range(len(labels)) if labels[i] == ci and pred[i] != ci]
                 for ci, c in enumerate(classes)}
for c, idx in misclassified.items():
    print(f"true '{c}' predicted as something else: {len(idx)}")
    if not idx:
        continue
    cols = 6; rows = (len(idx) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows), squeeze=False); axs = axs.ravel()
    for a in axs: a.axis("off")
    for a, i in zip(axs, idx):
        a.imshow(cv2.cvtColor(cv2.imread(val_paths[i]), cv2.COLOR_BGR2RGB))
        a.set_title(f"{val_paths[i].split('/')[-1][:18]}\\n-> {classes[pred[i]]} {probs[i, pred[i]]:.2f}", fontsize=8)
    fig.suptitle(f"true '{c}'", fontsize=11); plt.tight_layout(); plt.show()
"""),
md("""
## 5. Save & Download Pickle

The head as plain numbers: the scaler folded into its weights, so the Jetson computes
`softmax(weight · summary + bias)` and nothing else. The check computes exactly that on
the validation features and must give the probabilities above.
"""),
code("""
import pickle

scaler, lr = head[0], head[-1]
W = lr.coef_ / scaler.scale_
b = lr.intercept_ - (lr.coef_ * scaler.mean_ / scaler.scale_).sum(axis=1)
if W.shape[0] == 1:   # two classes: sklearn keeps one row; softmax([0, z]) == its probability
    W, b = np.vstack([np.zeros_like(W), W]), np.concatenate([[0.0], b])

z = X_val @ W.T + b
folded = np.exp(z - z.max(1, keepdims=True))
folded /= folded.sum(1, keepdims=True)
print(f"Folded head vs the pipeline: max probability difference {np.abs(folded - probs).max():.2e}")

try:
    colors = project.colors
    if not colors:
        raise AttributeError
except Exception:
    palette = ["#ff3333", "#33aaff", "#33cc66", "#ffcc00", "#aa33ff", "#ff8833", "#00c2a8", "#e85d75"]
    colors = {c: palette[i % len(palette)] for i, c in enumerate(classes)}

model_dict = {
    "backbone": BACKBONE,
    # plain lists, no numpy objects: the Jetson's numpy need not be Colab's
    "head": {"weight": W.astype(np.float32).tolist(), "bias": b.astype(np.float32).tolist()},
    "cls": classes,   # folder order = the head's row order
    "colors": colors,
    "meta": {
        "type": "cls",
        "arch": BACKBONE["name"] + "-linear",
        "license": "NVIDIA Open Model License",
    },
}

pickle_path = f"/content/{project_name}.pkl"
with open(pickle_path, "wb") as f:
    pickle.dump(model_dict, f)

print(f"Saved: {pickle_path}")
print(f"Total size: {os.path.getsize(pickle_path)/1024:.0f} KB")
print(f"Classes: {classes}")

try:
    from google.colab import files
    files.download(pickle_path)
except ImportError:
    pass
"""),
]

old["cells"] = cells
json.dump(old, open(path, "w"), indent=1, ensure_ascii=False)
open(path, "a").write("\n")
print("wrote", path, len(cells), "cells")
