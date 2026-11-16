"""Reference datasets for the Jetson training notebooks: small COCO val2017
subsets written in Roboflow's COCO export layout (train/valid/test, each
with _annotations.coco.json + images; a placeholder category at id 0, the
real classes from 1) — the layout train_segmentation / train_keypoint read.

usage: build_coco_subsets.py <coco annotations dir> <datasets dir>"""
import json, os, random, sys, urllib.request

ANN, OUT = sys.argv[1], sys.argv[2]
random.seed(0)
SPLITS = (("train", 0.7), ("valid", 0.2), ("test", 0.1))


def write(name, placeholder, categories, images, anns_by_image):
    root = os.path.join(OUT, name)
    random.shuffle(images)
    n, start = len(images), 0
    for i, (split, frac) in enumerate(SPLITS):
        stop = n if i == len(SPLITS) - 1 else start + round(n * frac)
        part = images[start:stop]; start = stop
        folder = os.path.join(root, split); os.makedirs(folder, exist_ok=True)
        out_images, out_anns = [], []
        for new_id, img in enumerate(part):
            path = os.path.join(folder, img["file_name"])
            if not os.path.exists(path):
                urllib.request.urlretrieve(img["coco_url"], path)
            out_images.append({"id": new_id, "file_name": img["file_name"], "width": img["width"], "height": img["height"]})
            for a in anns_by_image[img["id"]]:
                out_anns.append(dict(a, id=len(out_anns), image_id=new_id))
        json.dump({"info": {"description": "COCO val2017 subset (CC BY 4.0 annotations)"},
                   "categories": [placeholder] + categories, "images": out_images, "annotations": out_anns},
                  open(os.path.join(folder, "_annotations.coco.json"), "w"))
        print("%s/%s: %d images, %d annotations" % (name, split, len(out_images), len(out_anns)))


# ── SEG: bottle, cup, bowl — instance polygons ───────────────────────
coco = json.load(open(os.path.join(ANN, "instances_val2017.json")))
wanted = {c["id"]: c["name"] for c in coco["categories"] if c["name"] in ("bottle", "cup", "bowl")}
new_id = {old: i + 1 for i, old in enumerate(sorted(wanted))}
anns = {}
crowded = set()
for a in coco["annotations"]:
    if a["category_id"] in wanted:
        if a["iscrowd"]:
            crowded.add(a["image_id"])
        else:
            anns.setdefault(a["image_id"], []).append(
                {"category_id": new_id[a["category_id"]], "bbox": a["bbox"], "area": a["area"],
                 "segmentation": a["segmentation"], "iscrowd": 0})
images = [img for img in coco["images"] if img["id"] in anns and img["id"] not in crowded]
images = random.sample(images, 240)
write("coco_seg_sample", {"id": 0, "name": "objects", "supercategory": "none"},
      [{"id": new_id[old], "name": wanted[old], "supercategory": "objects"} for old in sorted(wanted)],
      images, anns)

# ── KP: people, every one with keypoints ─────────────────────────────
coco = json.load(open(os.path.join(ANN, "person_keypoints_val2017.json")))
person = coco["categories"][0]
people = {}
for a in coco["annotations"]:
    people.setdefault(a["image_id"], []).append(a)
anns, images = {}, []
for img in coco["images"]:
    ps = people.get(img["id"], [])
    if 1 <= len(ps) <= 4 and all(not p["iscrowd"] and p["num_keypoints"] >= 8 for p in ps):
        anns[img["id"]] = [{"category_id": 1, "bbox": p["bbox"], "area": p["area"], "keypoints": p["keypoints"],
                            "num_keypoints": p["num_keypoints"], "segmentation": [], "iscrowd": 0} for p in ps]
        images.append(img)
images = random.sample(images, 240)
write("coco_kp_sample", {"id": 0, "name": "people", "supercategory": "none"},
      [{"id": 1, "name": "person", "supercategory": "people",
        "keypoints": person["keypoints"], "skeleton": person["skeleton"]}],
      images, anns)
print("done")
