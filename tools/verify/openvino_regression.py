"""
The OpenVINO engine (the Pi) end to end: every model type it runs, on the
example models and images, through Detection — the path the server takes.

    python3 openvino_regression.py save  <results.json>   # record
    python3 openvino_regression.py check <results.json>   # compare

`check` passes when every case gives the same objects as the recording:
same classes, confidences within 1e-4, coordinates within 0.5 px. A kp
result is compared by its keypoints (name, center, conf), however they
are nested — a kp result is one entry per object with its keypoints, and
older recordings listed the keypoints flat.

Run from the repo root.
"""

import json
import sys

from dorna_vision.detect import Detection

M, I = "example/model/pi/", "example/img/"
CASES = {
    "od":      ({"cmd": "od", "path": M + "tube.pkl", "conf": 0.3}, I + "tube.jpg"),
    "rod":     ({"cmd": "rod", "path": M + "cardamom.pkl", "conf": 0.15}, I + "cardamom.jpg"),
    "kp":      ({"cmd": "kp", "path": M + "microplate_keypoint.pkl", "conf": 0.0, "kp_conf": 0.0}, I + "vial.jpg"),
    "anom":    ({"cmd": "anom", "path": M + "bottle_anomaly.pkl"}, I + "bottle_anomaly.png"),
    "ocr":     ({"cmd": "ocr", "conf": 0.3}, I + "ocr.jpg"),
    "barcode": ({"cmd": "barcode"}, I + "barcode.jpg"),
}


def _rows(cmd, results):
    """Each result as comparable rows: (kind, label, conf, points)."""
    rows = []
    for r in results:
        if cmd == "kp":
            for k in r.get("keypoints", [r]):     # nested (an object's), or a flat recording's
                rows.append(("kp", k["cls"], k["conf"], [k["center"]]))
            continue
        rows.append(("obj", r["cls"], r.get("conf"), [r["center"]] + list(r["corners"])))
    return sorted(rows, key=lambda row: (row[0], str(row[1]), row[3][0]))


def run_all():
    out = {}
    for name, (detection, image) in CASES.items():
        d = Detection(detection=detection)
        try:
            out[name] = _rows(detection["cmd"], d.run(data=image))
        finally:
            d.close()
    return out


def _same(a, b):
    if len(a) != len(b):
        return "%d results, recorded %d" % (len(a), len(b))
    for x, y in zip(a, b):
        if x[0] != y[0] or x[1] != y[1]:
            return "%s %r, recorded %s %r" % (x[0], x[1], y[0], y[1])
        if (x[2] is None) != (y[2] is None) or (x[2] is not None and abs(x[2] - y[2]) > 1e-4):
            return "%r conf %s, recorded %s" % (x[1], x[2], y[2])
        for p, q in zip(x[3], y[3]):
            if max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 0.5:
                return "%r point %s, recorded %s" % (x[1], p, q)
    return None


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    got = json.loads(json.dumps(run_all()))   # tuples → lists, as recorded
    if mode == "save":
        json.dump(got, open(path, "w"), indent=1)
        for name, rows in got.items():
            print("%-8s %d rows" % (name, len(rows)))
        sys.exit(0)
    recorded = json.load(open(path))
    failed = 0
    for name in CASES:
        problem = _same(got[name], recorded[name])
        print("%-8s %s" % (name, "ok (%d rows)" % len(got[name]) if problem is None else "FAIL: " + problem))
        failed += problem is not None
    sys.exit(1 if failed else 0)
