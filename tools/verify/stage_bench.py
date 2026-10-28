"""Accuracy first, then per-stage timings, for OD (RF-DETR) and CLS (DINOv3)
through dorna_vision.ai as the server runs them. One argument: a label
for the clock state. Prepares (builds) the engines when missing."""
import os, sys, time
import cv2
import numpy as np
from dorna_vision.ai import OD, CLS, prepare

label = sys.argv[1]
OD_P, CLS_P = "/home/dorna/scratch/od_rfdetr_small_coco.pkl", "/home/dorna/scratch/cls_nb/capped.pkl"
for p in (OD_P, CLS_P):
    r = prepare(p)
    if r["built"]:
        print("prepared %s in %.0fs" % (os.path.basename(p), r["seconds"]))
od, cls = OD(OD_P), CLS(CLS_P)

# accuracy
root = "/home/dorna/scratch/capped/valid"
ok = n = 0
for c in cls.cls:
    for f in sorted(os.listdir(f"{root}/{c}")):
        r = cls(cv2.imread(f"{root}/{c}/{f}"), conf=0.0)
        n += 1; ok += r[0][0] == c
ball = cv2.imread("/usr/share/opencv4/samples/data/basketball1.png")
print("[%s] CLS valid acc %.2f%% (%d wrong of %d)   OD basketball: %s" % (
    label, 100 * ok / n, n - ok, n, [(o.cls, round(o.prob, 3)) for o in od(ball, conf=0.5)]))


def ms(fn, reps=200):
    for _ in range(20):
        fn()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps * 1000


crop = cv2.imread(f"{root}/presence/" + sorted(os.listdir(f"{root}/presence"))[0])
for name, model, img in (("CLS", cls, crop), ("OD ", od, ball)):
    x = model._preprocess(img)
    pre = ms(lambda: model._preprocess(img))
    trt_ms = ms(lambda: model.model(x))
    full = ms(lambda: model(img))
    print("[%s] %s  preprocess %.2f ms   engine (copy in + infer + copy out) %.2f ms   full call %.2f ms   "
          "(post %.2f ms)" % (label, name, pre, trt_ms, full, full - pre - trt_ms))
