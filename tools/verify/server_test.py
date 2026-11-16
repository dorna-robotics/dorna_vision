"""
A running vision server, end to end, on either engine: each model is
prepared (a TensorRT model's engine built — then prepared again: nothing
to build), loaded by shipping its file as the playground does, run on an
image, and its results checked for the shape of its type.

    python3 server_test.py <host:port> <cmd>:<model.pkl>:<image> ...

Passing: every case prints "ok"; the server refuses a model of the other
engine (give one as "refuse:<model.pkl>:<image>").
"""

import sys
import time

from dorna_vision_client import VisionClient

KEYS = {"od": {"cls", "conf", "center", "corners"}, "rod": {"cls", "conf", "center", "corners"},
        "seg": {"cls", "conf", "center", "corners", "polygon"}, "kp": {"cls", "conf", "center", "corners", "keypoints"},
        "cls": {"cls", "conf", "center", "corners"}, "anom": {"cls", "conf", "center", "corners"}}

host, port = sys.argv[1].split(":")
c = VisionClient()
c.connect(host=host, port=int(port))
failed = 0
for i, case in enumerate(sys.argv[2:]):
    cmd, pkl, image = case.split(":")
    name = "t%d" % i
    if cmd == "refuse":
        try:
            c.detection_add(name=name, detection={"path": pkl}, timeout=120)
            print("%-6s FAIL: a model of the other engine was loaded" % cmd)
            failed += 1
        except Exception as ex:
            print("%-6s ok — refused: %s" % (cmd, str(ex)[:110]))
        continue
    t = time.time()
    first = c.model_prepare(pkl, timeout=3600)
    t1 = time.time() - t
    again = c.model_prepare(pkl)
    reply = c.detection_add(name=name, detection={"path": pkl}, timeout=120)
    t = time.time()
    results = c.detection(name).run(data=image)
    t2 = time.time() - t
    classes = c.detection(name).classes()
    c.detection_remove(name)
    problems = []
    if again["built"]:
        problems.append("prepared twice")
    if reply["cmd"] != cmd:
        problems.append("server read cmd %r" % reply["cmd"])
    if not results:
        problems.append("no results")
    for r in results:
        missing = KEYS[cmd] - set(r)
        if missing:
            problems.append("missing %s" % sorted(missing))
            break
        if cmd == "kp" and any(set(k) < {"cls", "conf", "center"} for k in r["keypoints"]):
            problems.append("a keypoint lacks cls/conf/center")
            break
    brief = [(r["cls"], round(r["conf"], 3)) + ((len(r["keypoints"]),) if cmd == "kp" else ())
             + ((len(r["polygon"]),) if cmd == "seg" else ()) for r in results[:4]]
    print("%-6s %s — engine %s, prepare %s in %.0f s, run %.2f s, %d results %s%s, classes %d" % (
        cmd, "ok" if not problems else "FAIL: " + "; ".join(problems), reply.get("engine"),
        "built" if first["built"] else "nothing to build", t1, t2, len(results), brief,
        "…" if len(results) > 4 else "", len(classes)))
    failed += bool(problems)
c.close()
sys.exit(1 if failed else 0)
