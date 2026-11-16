"""Parity, part 1 (rfdetr env): pretrained RF-DETR Small (OD), Seg-Small and
Keypoint pickles via rfdetr_pkl, and RF-DETR's own predict() on the test images."""
import json, os, pickle, sys
import cv2
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # tools/rfdetr_pkl.py
from rfdetr import RFDETRSmall, RFDETRSegSmall, RFDETRKeypointPreview
from rfdetr_pkl import rfdetr_model_dict

OUT = sys.argv[1]
# the scratch layout (tools/verify/README.md) — absolute: the checks run as root
SCRATCH = "/home/dorna/scratch"
IMG = SCRATCH + "/images/"   # OpenCV's sample images (4.8.0 samples/data)
IMAGES = [IMG + "basketball1.png", IMG + "basketball2.png",
          IMG + "fruits.jpg"]
coco_kp = json.load(open(sys.argv[2]))["categories"][0]          # person: keypoints + skeleton
refs = {}
for M, kind, export_dir in ((RFDETRSmall, "od", "od_export"), (RFDETRSegSmall, "seg", "seg_export"),
                            (RFDETRKeypointPreview, "kp", "kp_export")):
    model = M()
    onnx_path = model.export(output_dir=os.path.join(OUT, export_dir), verbose=False)
    extra = {}
    if kind == "kp":
        extra = {"keypoint_names": {"person": coco_kp["keypoints"]}, "skeleton": {"person": coco_kp["skeleton"]}}
    d = rfdetr_model_dict(model, onnx_path, kind, M.__name__, {}, **extra)
    pickle.dump(d, open(os.path.join(OUT, kind + ".pkl"), "wb"))
    print(kind, "slots named:", sum(c is not None for c in d["cls"]), "of", len(d["cls"]), {k: v for k, v in d["meta"].items() if k != "input"})
    for path in IMAGES:
        img = cv2.imread(path)
        det = model.predict(img[:, :, ::-1].copy(), threshold=0.5)
        rows = []
        for j in range(len(det.xy) if kind == "kp" else len(det.xyxy)):
            if kind == "od":
                rows.append({"cls": str(det.data["class_name"][j]),
                             "conf": float(det.confidence[j]), "box": det.xyxy[j].tolist()})
            elif kind == "seg":
                rows.append({"cls": str(det.data["class_name"][j]),
                             "conf": float(det.confidence[j]), "box": det.xyxy[j].tolist(),
                             "mask_area": int(det.mask[j].sum())})
                np.save(os.path.join(OUT, "mask_%s_%d.npy" % (os.path.basename(path), j)), det.mask[j])
            else:
                rows.append({"cls": str(det.data["class_name"][j]), "conf": float(det.detection_confidence[j]),
                             "box": det.data["xyxy"][j].tolist(), "xy": det.xy[j].tolist(),
                             "kp_conf": det.keypoint_confidence[j].tolist()})
        refs.setdefault(kind, {})[os.path.basename(path)] = rows
json.dump(refs, open(os.path.join(OUT, "ref.json"), "w"), indent=1)
print("part 1 done")
