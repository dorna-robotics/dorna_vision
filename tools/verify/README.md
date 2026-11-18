# Verification scripts

The checks that proved each engine's models — rerun them after any change
to `dorna_vision/ai/`, to a unit's OS (JetPack, Raspberry Pi OS), or to a
model's notebook. Each prints one line per result; what "passing" looks
like is in the table.

## Either engine

| script | checks | passing |
|---|---|---|
| `openvino_regression.py save\|check <results.json>` (a Pi, from the repo root) | every OpenVINO model type, OCR and barcodes on the example models and images, through `Detection` — against a recording made before a change | every case `ok` |
| `server_test.py <host:port> <cmd>:<model.pkl>:<image> … [refuse:<model.pkl>:x]` + a running server | each model prepared (then again: nothing to build), shipped as the playground ships it, run, its results the shape of its type; a file that is no model of this unit's engine refused | every case `ok`, exit 0 |

On the Pi: `openvino_regression.py` with the example models
(`example/model/pi/`); `server_test.py` against a server on a free port
(`sudo python3 -m dorna_vision.server --port 4011 --no-mqtt`).

## The Jetson (TensorRT)

They run on the Jetson from `/home/dorna/scratch` (scratch space, not part
of any install):

```
/home/dorna/scratch/
  images/       basketball1.png, basketball2.png, fruits.jpg — OpenCV 4.8.0 samples/data
  cls_capped/   the reference dataset (datasets/cls_capped, from the repo's dev machine)
  rfdetr-lib/   RF-DETR + CPU torch, below
  parity/       rfdetr_parity_ref.py's output: od.pkl, seg.pkl, kp.pkl, ref.json
  cls_nb/       run_notebook.py's CLS run: ref_cls.pkl
```

Some need RF-DETR / timm / torch, which the vision install does **not**
carry; those run from a throwaway folder, never the system:

```bash
mkdir -p ~/scratch/images && cd ~/scratch/images
for f in basketball1.png basketball2.png fruits.jpg; do
  curl -sfLO https://raw.githubusercontent.com/opencv/opencv/4.8.0/samples/data/$f; done
# RF-DETR, CPU torch — exports, references, CPU training checks
pip3 install --target ~/scratch/rfdetr-lib --extra-index-url https://download.pytorch.org/whl/cpu "rfdetr[train,onnx]==1.11.2"
# run with:  CUDA_VISIBLE_DEVICES="" PYTHONPATH=~/scratch/rfdetr-lib python3 -s ...
# (the unit's venv python — it sees the system packages; -s keeps ~/.local out)
```

The scripts under test run as the services do: `sudo env -i
PATH=/opt/dorna/venv/bin:/usr/bin:/bin HOME=/root python3 <script>` (engines
cache in `/root/.cache/dorna_vision/trt`). One heavy job at a time on the
Jetson — a TensorRT build next to a training run once ran it out of memory.

| script | checks | passing |
|---|---|---|
| `rfdetr_parity_ref.py <out> <coco_person_kp_categories.json>` (rfdetr env) then `rfdetr_parity_check.py <out>` (system) | pretrained RF-DETR Small, Seg-Small and Keypoint through our TensorRT OD / SEG / KP vs RF-DETR's own `predict()` | 0 class mismatches; boxes ≤ 1 px; mask IoU ≥ 0.99; keypoints ≤ 1 px |
| `run_notebook.py <notebook> <work> <dataset> <od\|seg\|kp\|cls> '<overrides json>'` (rfdetr env) then, for RF-DETR, `nb_parity_check.py <work> <dataset>/test <seg\|kp>` (system) | a TensorRT training notebook end to end on a reference dataset, then the fine-tuned model through TensorRT vs its own `predict()` | the notebook's ONNX check agrees; 0 class mismatches |
| `stage_bench.py <label>` (system) | OD (RF-DETR-S COCO) + CLS (capped) accuracy, then per-stage timings (its CLS figures are the DINOv3 classifier's) | CLS 99.28 % on capped valid; OD finds the basketball players; times in docs/ml-engines.md |
| `cls_bench.py <capped dir>` (a CUDA-torch env with timm) | the classifier backbone choice: DINOv3 sizes vs MobileNetV4-Small | DINOv3 ViT-S/16 frozen + linear: 99.28 % |
| `cls_valid.py <model.pkl> <dataset split dir>` (system, after `run_notebook.py ... cls`) | a classifier .pkl through the unit's CLS — the built-in backbone (`dorna_vision.ai.fetch()` first) and the head | the notebook's own valid accuracy, exactly (98.19 % on `cls_capped`); ~7 ms a call |

`coco_person_kp_categories.json`: a COCO annotation file whose first
category is the person with its keypoints and skeleton —
`datasets/kp_coco_sample/train/_annotations.coco.json` is one (its
category `0` is Roboflow's placeholder: pass a file with the person first,
`{"categories": [<the person category>]}`).
