import time
from dorna_vision_client import VisionClient
c = VisionClient(); c.connect(host="127.0.0.1", port=4001)
for name, cmd, P in (("od1", "od", "/home/dorna/scratch/od_rfdetr_small_coco.pkl"),
                     ("cls1", "cls", "/home/dorna/scratch/cls_nb/capped.pkl")):
    t = time.time()
    try:
        c.detection_add(name=name, detection={"cmd": cmd, "path": P, "conf": 0.5}, timeout=60)
        print(name, "UNEXPECTED: detection_add succeeded unprepared")
    except Exception as ex:
        print(name, "1. unprepared detection_add refused after %.1fs: %s" % (time.time() - t, str(ex)[:70]))
    t = time.time(); r = c.model_prepare(P); print(name, "2. model_prepare ->", r, "wall %.0fs" % (time.time() - t))
    t = time.time(); r = c.model_prepare(P); print(name, "3. again ->", r, "wall %.1fs" % (time.time() - t))
    t = time.time(); r = c.detection_add(name=name, detection={"cmd": cmd, "path": P, "conf": 0.5}); print(name, "4. detection_add ->", r, "wall %.1fs" % (time.time() - t))
print("detections:", c.detection_list())
c.close()
