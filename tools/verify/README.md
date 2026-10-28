# Jetson verification scripts

The checks that proved the Jetson's models — rerun them after any change
to `dorna_vision/ai.py`, to JetPack, or to a model's notebook. Each prints
one line per result; what "passing" looks like is under each.

They run on the Jetson from `~/scratch` (scratch space, not part of any
install). Some need RF-DETR / timm / torch, which the vision install does
**not** carry; those run from throwaway folders, never the system:

```bash
mkdir -p ~/scratch && cd ~/scratch
# RF-DETR, CPU torch — exports, references, CPU training checks
pip3 install --target ~/scratch/rfdetr-lib "rfdetr[train,onnx]==1.11.2"
# run with:  CUDA_VISIBLE_DEVICES="" PYTHONPATH=~/scratch/rfdetr-lib python3 -s ...
```

The scripts under test run as the services do: `sudo env -i
PATH=/usr/local/bin:/usr/bin:/bin HOME=/root python3 <script>` (engines
cache in `/root/.cache/dorna_vision/trt`). One heavy job at a time on the
Jetson — a TensorRT build next to a training run once ran it out of memory.

| script | checks | passing |
|---|---|---|
| `rfdetr_parity_ref.py <out> <coco_person_kp_categories.json>` (rfdetr env) then `rfdetr_parity_check.py <out>` (system) | pretrained RF-DETR Seg-Small and Keypoint through our TensorRT SEG / KP vs RF-DETR's own `predict()` | 0 class mismatches; boxes ≤ 1 px; mask IoU ≥ 0.99; keypoints ≤ 1 px |
| `run_rfdetr_notebook.py <notebook> <work> <dataset> <seg\|kp\|od> '<overrides json>'` (rfdetr env) then `nb_parity_check.py <work> <dataset>/test <seg\|kp>` (system) | a training notebook end to end on a reference dataset slice, then the fine-tuned model through TensorRT vs its own `predict()` | notebook prints its ONNX check (boxes agree); 0 class mismatches |
| `stage_bench.py <label>` (system) | OD (RF-DETR-S COCO) + CLS (capped) accuracy, then per-stage timings | CLS 99.28 % on capped valid; OD finds the basketball players; see docs/jetson.md for times |
| `prepare_test.py` + a server on :4001 | `model_prepare` → `detection_add` through the live vision server | unprepared refused at once; prepare builds; second prepare instant |
| `server_seg_kp_test.py` + a server on :4001 | SEG / KP results through the server: `polygon`, `keypoints` | results carry the keys; drawing runs |
| `cls_bench.py <capped dir>` (a CUDA-torch env with timm) | the classifier backbone choice: DINOv3 sizes vs MobileNetV4-Small | DINOv3 ViT-S/16 frozen + linear: 99.28 % |

Paths inside `stage_bench.py`, `prepare_test.py`, `server_seg_kp_test.py`
name the `~/scratch` layout they were run in (test pickles, the capped
dataset copy) — edit them to where yours are.
