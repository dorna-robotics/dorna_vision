import json, time
from dorna_vision_client import VisionClient
c = VisionClient(); c.connect(host="127.0.0.1", port=4001)
IMG = "/usr/share/opencv4/samples/data/basketball1.png"
for name, cmd, P in (("seg1", "seg", "/home/dorna/scratch/parity/seg.pkl"), ("kp1", "kp", "/home/dorna/scratch/parity/kp.pkl")):
    print(name, "prepare", c.model_prepare(P))
    print(name, "add", c.detection_add(name=name, detection={"cmd": cmd, "path": P, "conf": 0.5}))
    t = time.time(); res = c.detection(name).run(data=IMG); dt = time.time() - t
    print(name, "run %.2fs, %d results" % (dt, len(res)))
    for r in res[:3]:
        brief = {k: (v if k not in ("polygon", "keypoints") else "%d items, first %s" % (len(v), json.dumps(v[0]) if v else "-"))
                 for k, v in r.items() if k != "timestamp"}
        print("   ", brief)
    img = c.detection(name).get_img() if hasattr(c.detection(name), "get_img") else None
c.close()
