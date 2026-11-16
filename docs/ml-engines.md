# The ML engines — OpenVINO (the Pi) and TensorRT (the Jetson)

One code base, one branch (`pro`), on two kinds of vision unit:

| engine | unit | install | runs | notebooks |
|---|---|---|---|---|
| **OpenVINO** | a Raspberry Pi — the CPU | `pip install -e ".[pi]"` (upgrade `vision_pro`) | OD, ROD, KP (top-down), CLS, ANOM, OCR | `training_notebooks/openvino/` |
| **TensorRT** | an NVIDIA Jetson Orin Nano Super — the GPU | `pip install -e ".[jetson]"` (upgrade `jetson`) | OD, SEG, KP, CLS, OCR | `training_notebooks/tensorrt/` |

Everything outside the model code — the server, the playground, the
cameras, ROI, drawing, sorting, poses, barcodes, VLM — is the same code on
both. This file is the map: how a model runs on each engine, the model
types and their results, training, setting up a Jetson, and status.

## Status (2026-11-16)

STATUS

## How a model runs

`dorna_vision/ai/` — `__init__.py` dispatches, `ov.py` is the OpenVINO
engine, `trt.py` the TensorRT engine, `common.py` what both share.

- **The model file says its engine.** An OpenVINO model carries its IR
  (`xml` / `bin`), a TensorRT model its ONNX graph (`onnx`); `ai.OD(path)`
  (and SEG, ROD, KP, CLS, ANOM) loads it on that engine, importing only
  that engine. A model the unit cannot run is refused with what it is —
  never guessed, never converted.
- **Prepare once per unit** — `model_prepare` (server command;
  `VisionClient.model_prepare(path)`; `dorna_vision.ai.prepare(path)`), the
  playground's Initialize does it first:
  - TensorRT: the engine is built for this GPU and TensorRT version and
    cached (`~/.cache/dorna_vision/trt`, root's for the services) — 1–3
    minutes, once. **Loading never builds**: an unprepared model is
    refused at once (`ModelNotPreparedError`) with what to run.
  - OpenVINO: nothing to build (it compiles as a model loads); prepare
    returns `{"built": false}`.
- **OCR is built in**, the unit's engine's: on OpenVINO the models ship
  in the package (`dorna_vision/model/ocr`); on TensorRT
  `dorna_vision.ai.fetch_ocr()` downloads PP-OCRv6 (pinned revision +
  sha256) and builds its engines — once per unit, the upgrade runs it.
- **One inference context per thread** on both engines (OpenVINO infer
  requests, TensorRT execution contexts), so the server's worker threads
  infer concurrently on one model.

### The TensorRT model pickle

```
{"onnx": bytes, "cls": [...], "colors": {...},
 "meta": {"type": "od"|"seg"|"kp"|"cls", "image_size": ...,
          "precision": "fp16"|"bf16"|"fp32",
          "input": {"format": "uint8-bgr", "channels": "rgb"|"bgr", "mean": [3], "std": [3]}
                 | {"format": "float"},
          "profile": {"min", "opt", "max"}        # variable-shape graphs only
          ...}}
```

- **Precision is the model's.** The training notebook writes it: `fp16`
  for RF-DETR, `bf16` for DINOv3 (its attention scores overflow FP16's
  range — every output turns NaN; BF16 has FP32's range).
- **The camera image goes straight in.** With `format: uint8-bgr` the
  engine takes OpenCV's uint8 BGR image; cast, channel order, mean/std and
  the layout run on the GPU (nodes prepended to the ONNX before TensorRT
  parses it). The CPU only resizes / letterboxes.

## The model types

The result of each type has the same shape on both engines (one dict per
object, original-image pixels):

| cmd | OpenVINO (Pi) | TensorRT (Jetson) | result |
|---|---|---|---|
| `od` | mmdetection | RF-DETR (nano…large) | `cls, conf, center, corners` |
| `seg` | — | RF-DETR Seg (nano…xlarge) | + `polygon`: the outline, `[[x, y], …]` |
| `rod` | mmrotate | — | `corners`: the rotated rectangle |
| `kp` | RTMPose, top-down | RF-DETR Keypoints (Preview) | + `keypoints`: `[{cls (its name), conf, center, xyz?}]`; `pose_kp(result["keypoints"], geometry)` |
| `cls` | image classifier | DINOv3 ViT-S/16 + linear head | one per image: `cls, conf` |
| `anom` | STFPM | — | one verdict per image: `cls` (pass/fail), `conf` (the score) |
| `ocr` | horizontal-text det + PP-OCR rec (built in) | PP-OCRv6 medium (built in) | one per text line: `cls` (the text), `conf`, `corners` |
| `barcode` | zxing-cpp | zxing-cpp | `cls` (the content), `format` |
| `vlm` | Gemini (cloud) | Gemini (cloud); Cosmos Reason 2 2B (local) next | docs/vision-guide.md §9 |

Notes:

- **KP** — one entry per object, its keypoints nested; `conf` filters the
  objects, `kp_conf` each object's keypoints, `cls` the classes. A
  **top-down** model (OpenVINO's RTMPose, `model.top_down`) looks in one
  box: `bbox` (4 corner points like the ROI's; empty = the whole image)
  and `padding` (1.25 — the training framing); its object is that box,
  the pickle's one class, scored as mmpose scores a given box (the mean
  confidence of the keypoints it reports). RF-DETR finds every object and
  its keypoints in one pass. The playground shows bbox/padding for a
  top-down model only and seeds `kp_conf` per engine (0.3 RTMPose, 0.5
  RF-DETR).
- **RF-DETR has no NMS** — DETR predicts a set. Its classes are the
  model's logit slots, named exactly as RF-DETR's `predict()` names them
  (`None` = a slot it never reports — a pretrained model's sparse COCO
  ids, a keypoint model's background slot). RF-DETR resizes the uint8
  image (as its training does), so borderline scores can differ from
  `predict()` by a few hundredths.
- **CLS on TensorRT** has nothing to tune: no augmentation (it did not
  help a frozen backbone), the head's regularisation picked by
  cross-validation on the training images. Chosen by benchmark
  (`tools/verify/cls_bench.py`): ViT-S/16 beat ViT-S+/B and ConvNeXt-T/S/B
  as a frozen backbone, and tied MobileNetV4-Small fine-tuned (2 errors
  of 276 each) on the capped set.
- **OCR on TensorRT** follows PaddleOCR's pipeline (its OCR.yaml
  defaults, its DB post-process, crop and reading order) except one
  deliberate change: each text line is read on its own (PaddleOCR batches
  six, padded to the widest), so one line's text never depends on the
  others.

## Training a model

Training runs in **Colab** (GPU), never on a unit: the notebooks download
a Roboflow export, train, export the engine's format, check it against
PyTorch, and download the `.pkl`. Then on the unit: `model_prepare` once,
then use it. Train for the unit's engine — `training_notebooks/openvino/`
for a Pi, `training_notebooks/tensorrt/` for a Jetson.

- The three RF-DETR notebooks are generated from shared cells so they
  cannot drift: edit `tools/make_rfdetr_notebooks.py` / `tools/rfdetr_pkl.py`,
  then `python3 tools/make_rfdetr_notebooks.py training_notebooks/tensorrt tools/rfdetr_pkl.py`.
  CLS: `python3 tools/make_classification_notebook.py training_notebooks/tensorrt/train_classification.ipynb`.
- **Reference datasets** live in `datasets/` (git-ignored —
  `.git/info/exclude` — not committed): `cls_capped`, `seg_coco_sample`,
  `kp_coco_sample`, with their own README. The COCO ones are rebuilt with
  `python3 tools/build_coco_subsets.py <coco annotations dir> datasets`.

## Licenses that ship with the models

- **DINOv3** (CLS on TensorRT): Meta's DINOv3 License — commercial use
  allowed; no military / weapons use; a copy ships with the models:
  `licenses/DINOv3-LICENSE.md`.
- **PP-OCRv6**, **RF-DETR**: Apache-2.0.
- **Cosmos Reason 2** (next): NVIDIA Open Model License — commercial use
  allowed; products using it must say "Built on NVIDIA Cosmos".

## Setting up a Jetson

The install is the `upgrade` repo's `jetson` branch (its README): JetPack
7.2.1 — CUDA 13.2, TensorRT 10.16, NVIDIA's OpenCV 4.8 — with every python
package in one venv, `/opt/dorna/venv`, first on every PATH (so `sudo
python3` / `sudo pip3` are the venv's). It installs the same branches as
the Pis (this repo's `pro`, workspace's `main`); only the install differs.

1. **JetPack 7.2.1** (L4T r39.2.1, Ubuntu 24.04) — there is no SD-card
   image: write NVIDIA's Jetson ISO (`jetsoninstaller-r39.2.1-…-arm64.iso`)
   to a real USB 3.0 flash drive with Balena Etcher (with verification — a
   microSD in a USB reader made the installer fail with `nv-update-initrd`
   exit 127); boot the Jetson from it (DisplayPort monitor + USB keyboard:
   Esc at the NVIDIA logo → Boot Manager → the USB drive); accept the
   firmware update; "Install Jetson ISO r39.2.1" onto the microSD (wiped
   first: diskpart `clean`); first boot: user `dorna`, password `dorna`,
   hostname `jetson`, wired Ethernet.
2. Passwordless sudo, as on the Pis:
   `echo "dorna ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/010_dorna-nopasswd && sudo chmod 440 /etc/sudoers.d/010_dorna-nopasswd`
3. The upgrade's first run (`upgrade` repo, `jetson` branch, its README):
   MAXN SUPER, the NVIDIA stack, the venv, librealsense (RSUSB + CUDA),
   path_planning, the cameras (RealSense, uEye, Hikrobot, OAK), this
   package with its `jetson` extra and the OCR models, the vision server
   on :4001, the orchestrator on :80, Jupyter on :8888.

## Checking it

`tools/verify` (its README): the Pi's engine end to end against a
recording (`openvino_regression.py`), a running server on either engine
(`server_test.py`), and the Jetson's models against RF-DETR's own
`predict()` and PyTorch.

## Next steps

NEXT
