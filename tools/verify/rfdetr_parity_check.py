"""Parity, part 2 (system python): our TensorRT SEG and KP against RF-DETR's
own predict() (part 1's ref.json + masks)."""
import json, os, sys, time
import cv2
import numpy as np
from dorna_vision.ai import SEG, KP, prepare

OUT = sys.argv[1]
ref = json.load(open(os.path.join(OUT, "ref.json")))
IMAGES = {"basketball1.png": "/usr/share/opencv4/samples/data/basketball1.png",
          "basketball2.png": "/usr/share/opencv4/samples/data/basketball2.png",
          "fruits.jpg": "/usr/share/opencv4/samples/data/fruits.jpg"}

for kind, Model in (("seg", SEG), ("kp", KP)):
    pkl = os.path.join(OUT, kind + ".pkl")
    r = prepare(pkl)
    print(kind, "prepare", r)
    model = Model(pkl)
    worst = {"score": 0.0, "box": 0.0, "iou": 1.0, "kp_px": 0.0}
    mismatched = 0
    for name, path in IMAGES.items():
        img = cv2.imread(path)
        got = model(img, conf=0.5) if kind == "seg" else model(img, conf=0.5, kp_conf=0.5)
        exp = ref[kind][name]
        if sorted(o.cls for o in got) != sorted(e["cls"] for e in exp):
            mismatched += 1
            print("  MISMATCH", name, [(o.cls, round(o.prob, 3)) for o in got], [(e["cls"], round(e["conf"], 3)) for e in exp])
            continue
        used = set()
        for o in got:   # pair each of ours with RF-DETR's nearest same-class box
            box = np.array([o.rect.x, o.rect.y, o.rect.x + o.rect.w, o.rect.y + o.rect.h])
            j = min((j for j, e in enumerate(exp) if e["cls"] == o.cls and j not in used),
                    key=lambda j: np.abs(np.array(exp[j]["box"]) - box).max())
            used.add(j); e = exp[j]
            worst["score"] = max(worst["score"], abs(o.prob - e["conf"]))
            worst["box"] = max(worst["box"], float(np.abs(np.array(e["box"]) - box).max()))
            if kind == "seg":
                ours = np.zeros(img.shape[:2], np.uint8)
                if o.polygon:
                    cv2.fillPoly(ours, [np.array(o.polygon, np.int32)], 1)
                theirs = np.load(os.path.join(OUT, "mask_%s_%d.npy" % (name, j))).astype(np.uint8)
                iou = (ours & theirs).sum() / max((ours | theirs).sum(), 1)
                worst["iou"] = min(worst["iou"], float(iou))
            else:
                xy = np.array(e["xy"]); kc = np.array(e["kp_conf"])
                names = model.keypoint_names["person"]
                for k in o.keypoints:
                    i = names.index(k.name)
                    worst["kp_px"] = max(worst["kp_px"], float(np.abs(xy[i] - [k.x, k.y]).max()))
                visible_theirs = int((kc >= 0.5).sum())
                if visible_theirs != len(o.keypoints):
                    print("  visible keypoints differ", name, visible_theirs, len(o.keypoints))
        print("  %-16s %s" % (name, [(o.cls, round(o.prob, 3)) for o in got]))
    img = cv2.imread(IMAGES["basketball1.png"])
    for _ in range(10):
        model(img)
    t = time.perf_counter()
    for _ in range(50):
        model(img)
    print("%s: images %d, class mismatches %d, worst score diff %.3f, worst box diff %.1f px, %s, full call %.1f ms"
          % (kind.upper(), len(IMAGES), mismatched, worst["score"], worst["box"],
             "worst mask IoU %.3f" % worst["iou"] if kind == "seg" else "worst keypoint diff %.1f px" % worst["kp_px"],
             (time.perf_counter() - t) / 50 * 1000))
