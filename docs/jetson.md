# dorna_vision on the Jetson

The `jetson` branch runs the vision server on an **NVIDIA Jetson Orin Nano
Super**, next to the workspace orchestrator — one computer instead of a
workspace Pi and a vision Pi — with every ML model on the GPU through
**TensorRT**. This file is the map: how it works, how to train and deploy
each model, how to set up a Jetson, and what is done vs still to do.

## Status (2026-10-28)

**Verified on the Jetson** (JetPack 6.2.1, L4T 36.4.7, MAXN SUPER):

- **OD** — RF-DETR. Matches RF-DETR's own `predict()` (FP16 vs FP32 within
  0.01 on every usable detection); a model fine-tuned through the notebook
  matches its own predictions (0 class mismatches, boxes ≤ 0.1 px).
  RF-DETR-S at 512: ~15 ms a call.
- **CLS** — DINOv3 ViT-S/16 + linear head. 99.28 % on the capped set through
  TensorRT, exactly as in PyTorch; ~6 ms a call.
- **SEG** — RF-DETR Seg. Pretrained and notebook-fine-tuned: 0 class
  mismatches, mask IoU ≥ 0.99; ~25–40 ms a call (+ ~2 ms per object outline).
- **KP** — RF-DETR Keypoints (pretrained Preview): keypoints ≤ 1 px of
  RF-DETR's; ~57 ms a call.
- `model_prepare` through the live server; SEG `polygon` / KP `keypoints`
  results through the server.

**Written, not yet verified** — do these first (see *Next steps*):

- The camera-image input moved into the ONNX graph (`_image_input`) after
  the checks above — every engine rebuilds, every check reruns.
- **OCR** — PP-OCRv6 medium, ported from PaddleOCR's own pipeline.
- **KP** fine-tuned through `train_keypoint.ipynb` (the run was cut off).
- Barcode reading (zxing-cpp) on the Jetson.

## How a model runs on the Jetson

A model is a **pickle** (`.pkl`) that carries a portable **ONNX** graph and
how to build and feed it (`dorna_vision/ai.py`, top of file):

```
{"onnx": bytes, "cls": [...], "colors": {...},
 "meta": {"type": "od"|"seg"|"kp"|"cls", "image_size": ...,
          "precision": "fp16"|"bf16"|"fp32",
          "input": {"format": "uint8-bgr", "channels": "rgb"|"bgr", "mean": [3], "std": [3]}
                 | {"format": "float"},
          "profile": {"min", "opt", "max"}        # variable-shape graphs only
          ...}}
```

- **Prepare once per unit.** A TensorRT engine is tied to the GPU and the
  TensorRT version that built it, so it is built on the Jetson, never
  shipped: `model_prepare` (server command; `VisionClient.model_prepare(path)`;
  `dorna_vision.ai.prepare(path)`) builds and caches it
  (`~/.cache/dorna_vision/trt`, root's for the services) — 1–3 minutes, once.
  **Loading never builds**: an unprepared model is refused at once
  (`ModelNotPreparedError`) with what to run. The playground's Initialize
  prepares before it loads.
- **Precision is the model's.** The training notebook writes it:
  `fp16` for RF-DETR, `bf16` for DINOv3 (its attention scores overflow
  FP16's range — every output turns NaN; BF16 has FP32's range).
- **The camera image goes straight in.** With `format: uint8-bgr` the engine
  takes OpenCV's uint8 BGR image; cast, channel order, mean/std and the
  layout run on the GPU (nodes prepended to the ONNX before TensorRT parses
  it). The CPU only resizes / letterboxes.
- **One execution context per thread**, so the server's worker threads infer
  concurrently on one engine.

## The model types

| cmd | model | notebook | result (one dict per object) | measured |
|---|---|---|---|---|
| `od` | RF-DETR (nano…large) | `train_object_detection.ipynb` | `cls, conf, center, corners` | RF-DETR-S 512: 15 ms |
| `seg` | RF-DETR Seg (nano…xlarge) | `train_segmentation.ipynb` | + `polygon`: the outline, `[[x, y], …]` image px | Seg-S 384: 25–40 ms |
| `kp` | RF-DETR Keypoints (Preview) | `train_keypoint.ipynb` | + `keypoints`: `[{cls (its name), conf, center, xyz?}]` — every object and its keypoints in one pass; `pose_kp(result["keypoints"], geometry)` | 57 ms |
| `cls` | DINOv3 ViT-S/16 + linear head | `train_classification.ipynb` | one per image: `cls, conf` | 6 ms |
| `ocr` | PP-OCRv6 medium (built in) | — | one per text line: `cls` (the text), `conf`, `corners` | not yet measured |
| `barcode` | zxing-cpp | — | `cls` (the content), `format` | |
| `vlm` | Gemini (cloud) today; Cosmos Reason 2 2B (local) next | — | see docs/vision-guide.md §9 | |

ROD and ANOM are not on the Jetson branch.

Notes:

- **RF-DETR has no NMS** — DETR predicts a set. Its classes are the model's
  logit slots, named exactly as RF-DETR's `predict()` names them (`None` = a
  slot it never reports — a pretrained model's sparse COCO ids, a keypoint
  model's background slot). RF-DETR resizes the uint8 image (as its training
  does), so borderline scores can differ from `predict()` by a few hundredths.
- **CLS** has nothing to tune: no augmentation (it did not help a frozen
  backbone), the head's regularisation picked by cross-validation on the
  training images. Chosen by benchmark (`tools/verify/cls_bench.py`): ViT-S/16
  beat ViT-S+/B and ConvNeXt-T/S/B as a frozen backbone, and tied MobileNetV4-Small
  fine-tuned (2 errors of 276 each) on the capped set.
- **OCR** follows PaddleOCR's pipeline (its OCR.yaml defaults, its DB
  post-process, crop and reading order) except one deliberate change: each
  text line is read on its own (PaddleOCR batches six, padded to the widest),
  so one line's text never depends on the others. The models are fetched
  once per unit — `dorna_vision.ai.fetch_ocr()`, pinned revision + sha256 —
  into `~/.cache/dorna_vision/ocr` and prepared.

## Training a model

Training runs in **Colab** (GPU), never on the Jetson: the notebooks under
`training_notebooks/` download a Roboflow export, train, export ONNX, check
the ONNX against PyTorch, and download the `.pkl`. Then on the unit:
`model_prepare` once, then use it.

- The three RF-DETR notebooks are generated from shared cells so they cannot
  drift: edit `tools/make_rfdetr_notebooks.py` / `tools/rfdetr_pkl.py`, then
  `python3 tools/make_rfdetr_notebooks.py training_notebooks tools/rfdetr_pkl.py`.
  CLS: `python3 tools/make_classification_notebook.py training_notebooks/train_classification.ipynb`.
- **Reference datasets** live in `datasets/` (git-ignored — `.git/info/exclude`
  — not committed): `cls_capped`, `seg_coco_sample`, `kp_coco_sample`, with
  their own README. The COCO ones are rebuilt with
  `python3 tools/build_coco_subsets.py <coco annotations dir> datasets`.

## Licenses that ship with the models

- **DINOv3** (CLS): Meta's DINOv3 License — commercial use allowed; no
  military / weapons use; a copy ships with the models: `licenses/DINOv3-LICENSE.md`.
- **PP-OCRv6**: Apache-2.0.
- **RF-DETR**: Apache-2.0.
- **Cosmos Reason 2** (next): NVIDIA Open Model License — commercial use
  allowed; products using it must say "Built on NVIDIA Cosmos".

## Setting up a Jetson

The install is the `upgrade` repo's `jetson` branch (its README). Today it
targets JetPack 6.2.1; the move to **JetPack 7.2.1** is the next step.

1. **JetPack 7.2.1** (L4T r39.2.1, Ubuntu 24.04) — there is no SD-card image
   any more: write NVIDIA's Jetson ISO
   (`jetsoninstaller-r39.2.1-…-arm64.iso`) to a ≥16 GB USB stick with Balena
   Etcher; boot the Jetson from it (DisplayPort monitor + USB keyboard: Esc at
   the NVIDIA logo → Boot Manager → the USB stick; or headless with a 3.3 V
   USB-to-TTL adapter on the button header: adapter TX → pin 3, RX → pin 4,
   GND → pin 7); accept the firmware update (Y); "Install Jetson ISO
   r39.2.1" onto the microSD; first boot: user `dorna`, password `dorna`,
   hostname `jetson`, wired Ethernet.
2. Passwordless sudo, as on the Pis:
   `echo "dorna ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/010_dorna-nopasswd && sudo chmod 440 /etc/sudoers.d/010_dorna-nopasswd`
3. The upgrade's first run (`upgrade` repo, `jetson` branch, its README):
   MAXN SUPER, the build tools, librealsense (RSUSB + CUDA), path_planning,
   the cameras (RealSense, uEye, Hikrobot), the vision server on :4001, the
   orchestrator on :80, Jupyter on :8888 — and, to add with the JetPack
   7.2.1 update, the OCR models fetched and prepared (`fetch_ocr()`).

## Next steps

In order — one heavy job on the Jetson at a time:

1. **JetPack 7.2.1** on the Jetson (above), then update the upgrade `jetson`
   branch for it: the OS check (L4T r39), Ubuntu 24.04 / Python 3.12 (pip
   needs `--break-system-packages` again), CUDA 13 (`cuda-python` 13.x),
   TensorRT 10.16; run it on the fresh unit — that run is its test.
2. **Re-verify every model** with `tools/verify` (the ONNX-level image input
   rebuilds all engines): OD, CLS, SEG, KP parity; `model_prepare`; server.
3. **OCR**: `fetch_ocr()` (the detector's 4000 × 4000 profile makes its first
   build long — measure it, and whether a smaller max is enough), then check
   against PaddleOCR's own output on real label photos.
4. **KP** fine-tuned through `train_keypoint.ipynb` (`tools/verify`).
5. **Barcodes**: zxing-cpp on real bench images (QR, DataMatrix, Code128…).
6. **Cosmos Reason 2 2B** as a local `vlm` backend (TensorRT Edge-LLM, INT4)
   next to Gemini; measure it on the capped set and on before/after capping
   pairs; its logged verdicts become training data for dedicated models.
7. Decide `jetson_clocks` at boot (steadier, ~1–2 ms faster a call; more heat).
8. The golden SD image: clean the unit (SSH host keys, machine id), clone the
   card, compress.
