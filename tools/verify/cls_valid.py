"""A classifier .pkl (train_classification.ipynb's, run by run_notebook.py) through the unit's
CLS — the built-in backbone's engine and the head — on a Roboflow folder split: its accuracy, to set
beside the notebook's own, and the time of one call.

usage: cls_valid.py <model.pkl> <dataset split dir>"""
import os
import sys
import time

import cv2
import numpy as np

from dorna_vision.ai import CLS, prepare

PKL, ROOT = sys.argv[1:3]
print("prepare:", prepare(PKL))
cls = CLS(PKL)
ok = n = 0
for c in cls.cls:
    for f in sorted(os.listdir(os.path.join(ROOT, c))):
        r = cls(cv2.imread(os.path.join(ROOT, c, f)), conf=0.0)
        ok += r[0][0] == c
        n += 1
img = cv2.imread(os.path.join(ROOT, cls.cls[0], sorted(os.listdir(os.path.join(ROOT, cls.cls[0])))[0]))
for _ in range(20):
    cls(img)
ts = []
for _ in range(200):
    t = time.perf_counter()
    cls(img)
    ts.append((time.perf_counter() - t) * 1000)
print("CLS on the unit: %.2f%% (%d wrong of %d); one call %.1f ms (median of 200)" % (100 * ok / n, n - ok, n, np.median(ts)))
