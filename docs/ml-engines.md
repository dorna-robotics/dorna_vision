# The ML engines — OpenVINO (the Pi) and TensorRT (the Jetson)

One code base, one branch (`pro`), on two kinds of vision unit:

| engine | unit | install | runs | notebooks |
|---|---|---|---|---|
| **OpenVINO** | a Raspberry Pi — the CPU | `pip install -e ".[pi]"` (upgrade `vision_pro`) | OD, ROD, KP (top-down), CLS, ANOM, OCR | `training_notebooks/pi/` |
| **TensorRT** | an NVIDIA Jetson Orin Nano Super — the GPU | `pip install -e ".[jetson]"` (upgrade `jetson`) | OD, SEG, KP, CLS, OCR | `training_notebooks/jetson/` |

Everything outside the model code — the server, the playground, the
cameras, ROI, drawing, sorting, poses, barcodes, VLM — is the same code on
both. This file is the map: how a model runs on each engine, the model
types and their results, training, setting up a Jetson, and status.

## Status (2026-11-16)

Verified after the merge of the Pi's and the Jetson's code (`tools/verify`):

**OpenVINO — the Pi** (Raspberry Pi 5, OpenVINO 2025.3): OD, ROD, KP, ANOM,
OCR and barcodes give exactly what `pro` gave before the merge
(`openvino_regression.py`; KP now nested per object); every model type
through a running server (`server_test.py`), a file that is no model
refused.

**TensorRT — the Jetson** (Orin Nano Super, MAXN SUPER, JetPack 7.2.1:
TensorRT 10.16.2, CUDA 13.2):

| | check | result | one call |
|---|---|---|---|
| OD | RF-DETR-S (COCO) vs RF-DETR's `predict()` | 0 class mismatches, boxes ≤ 1.0 px, scores ≤ 0.014 | 16–17 ms |
| SEG | RF-DETR Seg-S vs `predict()` | 0 class mismatches, boxes ≤ 0.8 px, mask IoU ≥ 0.993 | 30 ms |
| KP | RF-DETR Keypoint vs `predict()` | 0 class mismatches, boxes ≤ 0.7 px, keypoints ≤ 0.7 px | 53 ms |
| CLS | the notebook trained on `cls_capped`, its .pkl (11 KB) through TensorRT (`cls_valid.py`) | 99.28 % on valid (2 of 276 wrong) — the notebook's own result, exactly | 4.4 ms |
| OCR | PP-OCRv6 medium | example image "JAGUAR" 0.996; a 4-line label read exactly at 1600 × 900 and 4056 × 2281 | 55–155 ms a run |
| barcode | zxing-cpp | EAN-13, QR, DataMatrix, Code128 | |

Every type through the live server (`server_test.py`), an OpenVINO model
refused. Engine builds (once per model per unit): OD 3.0, SEG 3.3, KP
3.7 minutes; built in, once per unit (the upgrade): the classifier
backbone 1 minute, OCR 17 (detection 10.4, recognition 6.4). A
classifier has nothing of its own to build.

Not yet: KP fine-tuned through `train_keypoint.ipynb` (the RF-DETR OD and
SEG notebooks were run end to end on JetPack 6).

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
- **OCR and the classifier backbone are built in**, the unit's engine's:
  on OpenVINO the OCR models ship in the package (`dorna_vision/model/ocr`);
  on TensorRT `dorna_vision.ai.fetch()` downloads PP-OCRv6 and DINOv3
  ViT-S/16 (pinned + sha256) and builds their engines — once per unit, the upgrade
  runs it.
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
  for RF-DETR; the classifier backbone is `bf16` (ViT attention scores
  can overflow FP16's range — DINOv3's turned every output NaN; BF16 has
  FP32's range).
- **A classifier carries no graph.** Its backbone is frozen and the same
  in every classifier, so it is the unit's (one engine, shared by every
  classifier loaded); the pickle names it and carries the linear head:

  ```
  {"backbone": {"name": "dinov3_vits16", "repo": "timm/vit_small_patch16_dinov3.lvd1689m", "revision": ...},
   "head": {"weight": [[...]] (C x 384), "bias": [...]},   # plain lists
   "cls": [...], "colors": {...}, "meta": {"type": "cls", "arch": "dinov3_vits16-linear", ...}}
  ```

  Under 100 KB: `model_prepare` only checks the unit has that backbone, and a
  classifier trained on another release is refused with what to do. The
  backbone's ONNX (the normalised image in, its 384-d pooled features
  out) is exported once from the pinned Hugging Face release the notebook
  loads and hosted with its sha256 — a release of this repo
  (`backbone-dinov3_vits16`; `_BACKBONES` in `ai/trt.py`). One engine
  (~47 MB) serves every classifier loaded; a classifier adds its head.
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
| `cls` | image classifier | DINOv3 ViT-S/16 (built in) + linear head | one per image: `cls, conf` |
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
  help a frozen backbone), every class weighted equally (a few fail
  images next to many good ones otherwise pull the head toward good), the
  head's regularisation picked by cross-validation on the training
  images. The backbone was chosen by benchmark (2026-11-18, the clean
  capped splits: whole bursts kept together, and across two sessions two
  hours apart; the head trained on 20 or 50 bad images per kind). Share
  of 1,243 test images wrong, DINOv3 ViT-S/16 on the unit's engine:

  | the check | 20 bad per kind | 50 | all |
  |---|---|---|---|
  | missing cap = fail, half-seated = pass | **2.0 %** | 1.2 % | 1.0 % |
  | missing and half-seated = fail | 11.6 % | 9.2 % | 9.1 % |

  The errors left in the second row are half-seated caps across
  sessions: a millimetre or two in a small crop — the image carries too
  little signal for any frozen backbone (C-RADIOv3-B, also tried: 8.4 %,
  but worse on missing caps, 9 % of them passed against DINOv3's 4 %,
  and slower). The head's cutoff trades the two errors: on the missing-cap
  check, P(pass) ≥ 0.9 passes 0.9 % of missing caps and pauses 3 % of
  good ones. The head trained on the notebook's PyTorch float32 features
  and run on the unit's BF16 engine gives the same verdicts within a
  tenth of a point. Earlier (`tools/verify/cls_bench.py`, the capped
  set's random split): ViT-S/16 beat ViT-S+/B and ConvNeXt-T/S/B and
  tied MobileNetV4-Small fine-tuned.
- **OCR on TensorRT** follows PaddleOCR's pipeline (its OCR.yaml
  defaults, its DB post-process, crop and reading order) except two
  deliberate changes: text detection runs on the image scaled to at most
  960 on its longest side — PP-OCRv4's size (PP-OCRv6 keeps up to 4000: its
  engine took over an hour to build on the Orin Nano and detects slower;
  960 reads a label the camera frames whole, a 4K frame included); and
  each text line is read on its own (PaddleOCR batches six, padded to the
  widest), so one line's text never depends on the others. The cache
  (`~/.cache/dorna_vision/ocr`) holds only the download; how each model is
  built is the code's.

## Training a model

Training runs in **Colab** (GPU), never on a unit: the notebooks download
a Roboflow export, train, export the engine's format, check it against
PyTorch, and download the `.pkl`. Then on the unit: `model_prepare` once,
then use it. Train for the unit's engine — `training_notebooks/pi/`
for a Pi, `training_notebooks/jetson/` for a Jetson.

- The three RF-DETR notebooks are generated from shared cells so they
  cannot drift: edit `tools/make_rfdetr_notebooks.py` / `tools/rfdetr_pkl.py`,
  then `python3 tools/make_rfdetr_notebooks.py training_notebooks/jetson tools/rfdetr_pkl.py`.
  CLS: `python3 tools/make_classification_notebook.py training_notebooks/jetson/train_classification.ipynb`.
- **Reference datasets** live in `datasets/` (git-ignored —
  `.git/info/exclude` — not committed): `cls_capped`, `seg_coco_sample`,
  `kp_coco_sample`, with their own README. The COCO ones are rebuilt with
  `python3 tools/build_coco_subsets.py <coco annotations dir> datasets`.

## Licenses that ship with the models

- **DINOv3** (the classifier backbone on TensorRT): Meta's DINOv3
  License — commercial use allowed; no military / weapons use; a copy
  ships with the models: `licenses/DINOv3-LICENSE.md` (and with the
  backbone's release).
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
   package with its `jetson` extra and its built-in models (OCR, the
   classifier backbone), the vision server
   on :4001, the orchestrator on :80, Jupyter on :8888.

## Checking it

`tools/verify` (its README): the Pi's engine end to end against a
recording (`openvino_regression.py`), a running server on either engine
(`server_test.py`), and the Jetson's models against RF-DETR's own
`predict()` and PyTorch.

## Next steps

1. **KP** fine-tuned through `training_notebooks/jetson/train_keypoint.ipynb`
   (`run_notebook.py` + `nb_parity_check.py`).
2. **Cosmos Reason 2 2B** as a local `vlm` backend on the Jetson
   (TensorRT Edge-LLM, INT4) next to Gemini; measure it on the capped set
   and on before/after capping pairs; its logged verdicts become training
   data for dedicated models.
3. Decide `jetson_clocks` at boot (steadier, ~1–2 ms faster a call; more heat).
4. The golden SD image: clean the unit (SSH host keys, machine id), clone
   the card, compress.
