"""Run a training notebook's own code cells (not Roboflow's download, not
pip) on a local dataset, with test-size overrides; then save RF-DETR's own
predictions on the test split for the Jetson parity check.

usage: run_rfdetr_notebook.py <notebook> <workdir> <dataset dir> <task> <overrides json>"""
import json, os, sys, types

nb_path, WORK, DATA, TASK, overrides = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], json.loads(sys.argv[5])
os.makedirs(WORK, exist_ok=True)

g = {"dataset": types.SimpleNamespace(location=DATA),
     "project": types.SimpleNamespace(name="ref_" + TASK, colors={})}

nb = json.load(open(nb_path))
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "Roboflow(" in src:
        continue
    src = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("!"))   # shell lines
    src = src.replace('"/content/', '"%s/' % WORK)
    src = src.replace("from google.colab import files",
                      "files = __import__('types').SimpleNamespace(download=lambda p: print('download:', p))")
    exec(compile(src, "%s cell %d" % (nb_path, nb["cells"].index(cell)), "exec"), g)
    if "MODEL_SIZE" in g and "_overridden" not in g:          # right after the parameters cell
        g.update(overrides); g["_overridden"] = True
        g["Model"] = g["MODELS"][g["MODEL_SIZE"]]
        print("overrides:", overrides)

# RF-DETR's own predictions on the test split, for the parity check
import cv2
best = g["best"]
refs = {}
test_dir = os.path.join(DATA, "test")
for f in sorted(os.listdir(test_dir)):
    if not f.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img = cv2.imread(os.path.join(test_dir, f))
    det = best.predict(img[:, :, ::-1].copy(), threshold=0.3)
    rows = []
    if TASK == "kp":
        for j in range(len(det.xy)):
            rows.append({"cls": str(det.data["class_name"][j]), "conf": float(det.detection_confidence[j]),
                         "box": det.data["xyxy"][j].tolist(), "xy": det.xy[j].tolist(),
                         "kp_conf": det.keypoint_confidence[j].tolist()})
    else:
        for j in range(len(det.xyxy)):
            row = {"cls": str(det.data["class_name"][j]), "conf": float(det.confidence[j]), "box": det.xyxy[j].tolist()}
            if TASK == "seg":
                import numpy as np
                np.save(os.path.join(WORK, "mask_%s_%d.npy" % (f, j)), det.mask[j])
            rows.append(row)
    refs[f] = rows
json.dump(refs, open(os.path.join(WORK, "ref.json"), "w"), indent=1)
print("notebook run done:", TASK, "test images", len(refs), "detections", sum(len(v) for v in refs.values()))
