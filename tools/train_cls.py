"""
Train a classifier on this Jetson — for internal use. Customers train on Colab
(training_notebooks/jetson/train_classification.ipynb); this is the same training
run on the unit, its features from the unit's own engine, so it needs no Colab, no
GPU training and no upload: seconds for a thousand images.

Every step is the notebook's: the image through the built-in backbone exactly as CLS
feeds it (centred letterbox, the engine), a linear head — standardised features,
logistic regression, every class weighted equally, its regularisation picked by
5-fold cross-validation on the training images — folded into weight · features +
bias, checked against the pipeline, and written as the notebook's .pkl.

    sudo python3 tools/train_cls.py <dataset dir> <out.pkl>

The dataset is a Roboflow "folder" export (or the same layout by hand):
<dataset>/train/<class>/<image> and, optional, <dataset>/valid/<class>/<image> —
scored, never trained on. The classes are train's folders, sorted. The backbone must
be prepared on this unit (dorna_vision.ai.fetch(), the upgrade runs it).
"""

import os
import pickle
import sys
import time

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dorna_vision.ai import trt

BACKBONE = "dinov3_vits16"      # the backbone training_notebooks/jetson/train_classification.ipynb uses
IMAGES = (".jpg", ".jpeg", ".png", ".bmp")
PALETTE = ["#ff3333", "#33aaff", "#33cc66", "#ffcc00", "#aa33ff", "#ff8833", "#00c2a8", "#e85d75"]


def load_split(root, classes):
    paths, labels = [], []
    for ci, c in enumerate(classes):
        folder = os.path.join(root, c)
        if os.path.isdir(folder):
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith(IMAGES):
                    paths.append(os.path.join(folder, f))
                    labels.append(ci)
    return paths, np.array(labels)


def features(model, size, paths):
    return np.stack([model(trt._letterbox_centered(cv2.imread(p), size)[None])["features"].reshape(-1).copy()
                     for p in paths])


def main(data, out):
    spec = trt._BACKBONES[BACKBONE]
    train_root = os.path.join(data, "train")
    classes = sorted(d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d)))
    if len(classes) < 2:
        raise SystemExit("%s: needs a folder per class (two at least), found %s" % (train_root, classes))
    train_paths, y_train = load_split(train_root, classes)
    val_paths, y_val = load_split(os.path.join(data, "valid"), classes)
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train: {len(train_paths)} images (per class: {np.bincount(y_train, minlength=len(classes)).tolist()})")
    print(f"Valid: {len(val_paths)} images")

    t = time.time()
    model = trt._backbone_acquire(BACKBONE)
    try:
        X_train = features(model, spec["image_size"], train_paths)
        X_val = features(model, spec["image_size"], val_paths) if val_paths else None
    finally:
        trt._backbone_release(BACKBONE)
    print(f"Features: {X_train.shape[1]} per image, {len(train_paths) + len(val_paths)} images in {time.time() - t:.1f} s")

    # the notebook's head: standardised features, logistic regression with every class weighted
    # equally (a few fail images next to many good ones otherwise pull it toward good), its
    # regularisation C by 5-fold cross-validation on train only. Newton-CG: lbfgs, fit cold at
    # each C, failed its line search at large C on separable features
    head = make_pipeline(StandardScaler(), GridSearchCV(
        LogisticRegression(solver="newton-cg", max_iter=5000, class_weight="balanced"),
        {"C": np.logspace(-4, 4, 10)}, cv=5, scoring="accuracy"))
    head.fit(X_train, y_train)
    print(f"Train acc: {100 * head.score(X_train, y_train):.2f}%  (C {head[-1].best_params_['C']:.3g})")

    scaler, lr = head[0], head[-1].best_estimator_
    W = lr.coef_ / scaler.scale_
    b = lr.intercept_ - (lr.coef_ * scaler.mean_ / scaler.scale_).sum(axis=1)
    if W.shape[0] == 1:   # two classes: sklearn keeps one row; softmax([0, z]) == its probability
        W, b = np.vstack([np.zeros_like(W), W]), np.concatenate([[0.0], b])

    if X_val is not None:
        probs = head.predict_proba(X_val)
        pred = probs.argmax(1)
        print(f"Valid acc: {100 * (pred == y_val).mean():.2f}%  ({int((pred != y_val).sum())} wrong of {len(y_val)})")
        print("rows = true", classes, "| cols = predicted", classes)
        print(confusion_matrix(y_val, pred, labels=list(range(len(classes)))))
        z = X_val @ W.T + b
        folded = np.exp(z - z.max(1, keepdims=True))
        folded /= folded.sum(1, keepdims=True)
        print(f"Folded head vs the pipeline: max probability difference {np.abs(folded - probs).max():.2e}")
        if "pass" in classes and "fail" in classes:
            i_pass, i_fail = classes.index("pass"), classes.index("fail")
            print("t      missed fails   good parts rejected")
            for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
                p = np.where(probs[:, i_pass] >= t, i_pass, i_fail)
                cm = confusion_matrix(y_val, p, labels=[i_fail, i_pass])
                print(f"{t:<6} {cm[0, 1]:>8}        {cm[1, 0]:>8}")

    model_dict = {
        "backbone": {"name": BACKBONE, **spec["source"]},
        # plain lists, no numpy objects: the pickle loads whatever the reader's numpy
        "head": {"weight": W.astype(np.float32).tolist(), "bias": b.astype(np.float32).tolist()},
        "cls": classes,   # folder order = the head's row order
        "colors": {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(classes)},
        "meta": {"type": "cls", "arch": BACKBONE + "-linear", "license": "DINOv3 License (Meta)"},
    }
    with open(out, "wb") as f:
        pickle.dump(model_dict, f)
    print(f"Saved: {out} ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
