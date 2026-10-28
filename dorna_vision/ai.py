"""
ML inference primitives for the dorna_vision pipeline (Jetson).

Each class is a single-purpose, init-once / call-many wrapper around a
model packaged in a pickle. On the Jetson the models run on the GPU with
TensorRT: the pickle carries a portable ONNX graph and the precision to
build it at (meta["precision"]), and prepare() builds the TensorRT engine
from it on this computer, once per model (an engine is tied to the GPU
and the TensorRT version that built it), and caches it. Loading a model never builds — an unprepared one raises
ModelNotPreparedError. The pickle schema (produced by the training
notebooks under training_notebooks/) is:

    {
        "onnx":    <ONNX graph, bytes>,
        "cls":     ["class_a", "class_b", ...]      # for OD/ROD/CLS/ANOM
        "colors":  {"class_a": "#aabbcc", ...}      # dict OR list of hex
        "meta":    {"image_size": ..., "precision": "fp16"|"bf16"|"fp32", ...},

        # KP-only:
        "keypoint_names": [...],
        "skeleton":       [(1,2), ...]
    }

    OD   — RF-DETR (TensorRT)
    CLS  — DINOv3 ViT-S/16 + a linear head (TensorRT)
    ROD, KP, ANOM, OCR — still OpenVINO IR ("xml"/"bin" in the
    pickle) until each moves to TensorRT; OpenVINO is imported only when
    one of them loads, and is not installed on the Jetson.

Composition: OD/ROD give you boxes; crop the image around a box (with
padding) and feed the crop to KP. CLS labels a whole image (or a crop).
ANOM scores a whole image and returns hot-spots.
"""

import hashlib
import math
import os
import pickle
import importlib
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

from dorna_vision.visual import *
from dorna_vision.util import *


# ───────────────────────────────────────────────────────────────────────
# Shared helpers
# ───────────────────────────────────────────────────────────────────────

# No explicit config — let OpenVINO pick defaults appropriate for the host
# (uses every physical core, runtime-chosen streams + LATENCY hint on
# recent versions). Override per call by passing `device_config=` to any
# of the inference classes.
_DEVICE_CONFIG = {}

# OD / ROD use the BGR-order detector mean/std (mmdet convention).
_DET_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
_DET_STD  = np.array([57.375, 57.12, 58.395], dtype=np.float32)

# CLS / KP / ANOM use ImageNet RGB-order mean/std, scaled to 0..255.
_IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
_IMNET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255

_PAD_VALUE = 128


def _hex_to_bgr(hex_color):
    h = str(hex_color).lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)
    return (0, 0, 255)


def _color_for_class(cid, class_names, colors):
    """
    Resolve a BGR color tuple for a class id.

    `colors` may be a dict {class_name: hex} (Roboflow / new training default)
    or a positional list [hex, hex, ...] (legacy). Falls back to red.
    """
    name = class_names[int(cid)]
    if isinstance(colors, dict):
        return _hex_to_bgr(colors.get(name, "#ff3333"))
    if isinstance(colors, (list, tuple)) and len(colors) > 0:
        return _hex_to_bgr(colors[int(cid) % len(colors)])
    return _hex_to_bgr("#ff3333")


def _letterbox(img, size):
    """
    Resize proportionally and pad to a square `size x size` with a constant
    grey. Returns (padded_image, ratio). Top-left aligned (matches the
    training notebooks under draft/).
    """
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((size, size, 3), _PAD_VALUE, dtype=np.uint8)
    padded[:nh, :nw] = resized
    return padded, r


def _letterbox_centered(img, size):
    """
    Resize proportionally and centre-pad to (W, H). Used by CLS where the
    classifier was trained with centre-padded inputs.
    """
    h, w = img.shape[:2]
    tw, th = size
    r = min(tw / w, th / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((th, tw, 3), _PAD_VALUE, dtype=np.uint8)
    pw = (tw - nw) // 2
    ph = (th - nh) // 2
    padded[ph:ph + nh, pw:pw + nw] = resized
    return padded


def _load_pickle(path):
    """
    Load a model pickle. Accepts either a filesystem path or a pre-loaded
    dict.
    """
    if isinstance(path, dict):
        return path
    with open(path, "rb") as f:
        return pickle.load(f)


class _PerThreadModel:
    """A compiled OpenVINO model that infers on ONE InferRequest PER THREAD.

    ``CompiledModel.__call__`` builds a request the first time and then
    REUSES that same one for every later call (its own docstring says so
    and advises a dedicated request instead), while a running request
    blocks every method on itself. The vision server answers each
    websocket call on a thread-pool worker, so two detections against the
    same model overlap as a matter of course — and the second one died
    with ``RuntimeError: Infer Request is busy``.

    A request is the unit of execution, so each thread gets its own, made
    on first use and released when the thread ends. Inference stays
    concurrent: no lock, no queue, no serialising the pool down to one.
    Everything else (``input``, ``output``, ``outputs``, …) passes
    straight through, so callers cannot tell the difference.
    """

    __slots__ = ("_compiled", "_local")

    def __init__(self, compiled):
        self._compiled = compiled
        self._local = threading.local()

    def __call__(self, inputs=None, share_inputs=True, share_outputs=False,
                 *, decode_strings=True):
        request = getattr(self._local, "request", None)
        if request is None:
            request = self._compiled.create_infer_request()
            self._local.request = request
        return request.infer(inputs, share_inputs=share_inputs,
                             share_outputs=share_outputs,
                             decode_strings=decode_strings)

    def __getattr__(self, name):
        return getattr(self._compiled, name)


def _read_openvino_from_pickle(model_dict, input_shape, device_name="CPU", device_config=None):
    """
    Build a compiled OpenVINO model from {xml, bin} pickle entries.
    `input_shape` is a list passed to model.reshape(). Returns
    (core, model, compiled_model).

    Validates the pickle is the current OpenVINO format before handing the
    blobs to the OpenVINO runtime. The old training-output format embedded a
    raw PyTorch state_dict under meta["weight"] and lacked real IR in
    xml/bin; passing that through could crash the native runtime. We reject it
    with a clear, actionable message instead.
    """
    # Old-format detector: a raw PyTorch checkpoint under meta.weight.
    meta = model_dict.get("meta") or {}
    if "weight" in meta:
        raise ValueError(
            "Unsupported model format: this looks like an old training-output "
            "pickle (raw PyTorch weights under meta['weight']), not an "
            "OpenVINO model. Re-export it from the current training notebook "
            "to produce an OpenVINO pickle."
        )

    xml_data = model_dict.get("xml")
    bin_data = model_dict.get("bin")
    if xml_data is None or bin_data is None:
        raise ValueError(
            "Invalid model: missing OpenVINO IR ('xml'/'bin' entries). "
            "Re-export the model from the current training notebook."
        )

    from openvino import Core   # the models not yet on TensorRT only
    core = Core()
    try:
        model = core.read_model(
            model=bytes(xml_data, "utf-8") if isinstance(xml_data, str) else xml_data,
            weights=bin_data,
        )
        if input_shape is not None:
            model.reshape({model.input(0): input_shape})
        cfg = device_config if device_config is not None else _DEVICE_CONFIG
        compiled = core.compile_model(model=model, device_name=device_name, config=cfg)
    except ValueError:
        raise
    except Exception as ex:
        # Any failure inside the OpenVINO runtime (malformed IR, bad shapes,
        # corrupt blob) -> a clean Python error the server can report, never
        # a native crash that drops the connection.
        raise ValueError("Failed to load OpenVINO model: %s" % ex)
    return core, model, _PerThreadModel(compiled)


def _release_openvino(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass


# ───────────────────────────────────────────────────────────────────────
# TensorRT — the Jetson GPU
# ───────────────────────────────────────────────────────────────────────

# Engines built from a pickle's ONNX, cached by the ONNX bytes + the
# TensorRT version + the GPU + the build settings: an engine runs only on
# the GPU and the TensorRT that built it, so it is built here, never shipped.
_TRT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "dorna_vision", "trt")

# The builder's scratch memory while it times candidate kernels. The GPU
# shares the Jetson's 8 GB with everything else; unbounded, the builder
# grabs what happens to be free and skips the kernels that did not fit —
# the engine would depend on what else was running. Fixed, it does not.
_TRT_WORKSPACE = 1 << 30

# One build at a time: a build holds the GPU for minutes.
_TRT_BUILD_LOCK = threading.Lock()

_TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def _cuda(result):
    """Unwrap a cuda-python call: raise on an error, return the value(s)."""
    err, *rest = result
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError("CUDA: %s" % cudart.cudaGetErrorString(err)[1].decode())
    return rest[0] if len(rest) == 1 else None


class ModelNotPreparedError(ValueError):
    """The model's TensorRT engine is not built on this computer yet."""


# The precisions a model may declare (pickle meta["precision"]), as the
# builder flag each sets; "fp32" sets none. The training notebook picks
# it for its architecture: FP16 where the activations fit its range
# (RF-DETR), BF16 — FP32's range, half the bits, as fast on Orin — where
# they do not: DINOv3's attention scores (Q·Kᵀ, queries and keys in the
# hundreds) overflow FP16's 65504 and every output turns NaN.
_TRT_PRECISIONS = {"fp16": trt.BuilderFlag.FP16, "bf16": trt.BuilderFlag.BF16, "fp32": None}


def _trt_precision(data):
    """The model's declared engine precision (pickle meta["precision"])."""
    precision = (data.get("meta") or {}).get("precision")
    if precision not in _TRT_PRECISIONS:
        raise ValueError(
            "Invalid model: meta['precision'] must be one of %s (got %r) — "
            "re-export it from its training notebook." % (sorted(_TRT_PRECISIONS), precision)
        )
    return precision


def _trt_engine_path(onnx_bytes, precision):
    """Where the engine for this ONNX graph at this precision lives in the cache."""
    prop = _cuda(cudart.cudaGetDeviceProperties(0))
    key = hashlib.sha256()
    key.update(onnx_bytes)
    key.update(("%s|%s|%d.%d|%s|%d|uint8-bgr-nhwc" % (trt.__version__, prop.name.decode(), prop.major,
                                                       prop.minor, precision, _TRT_WORKSPACE)).encode())
    return os.path.join(_TRT_CACHE, key.hexdigest()[:16] + ".engine")


def _image_input(network):
    """
    Give the engine the camera's image as its input: uint8, (1, H, W, 3),
    BGR — what OpenCV hands us. The model's own input — float (1, 3, H, W),
    RGB, ImageNet mean/std, every ONNX model's contract here — is computed
    from it on the GPU: cast, BGR→RGB, x·(1/255σ) − μ/σ, NHWC→NCHW. The CPU
    only resizes / letterboxes (OpenCV, uint8), and a quarter of the bytes
    cross to the GPU. Returns the constant arrays TensorRT reads at build
    time — they must live until the engine is built.
    """
    model_input = network.get_input(0)
    n, c, h, w = model_input.shape
    if network.num_inputs != 1 or c != 3 or model_input.dtype != trt.float32:
        raise ValueError("Invalid ONNX model: one float input (1, 3, H, W) expected, got %s %s"
                         % (model_input.dtype, tuple(model_input.shape)))
    order = np.array([2, 1, 0], dtype=np.int32)
    scale = (1.0 / (255.0 * np.array([0.229, 0.224, 0.225], dtype=np.float32))).reshape(1, 1, 1, 3)
    shift = (-np.array([0.485, 0.456, 0.406], dtype=np.float32)
             / np.array([0.229, 0.224, 0.225], dtype=np.float32)).reshape(1, 1, 1, 3)

    image = network.add_input("image", trt.uint8, (n, h, w, 3))
    x = network.add_cast(image, trt.float32).get_output(0)
    x = network.add_gather(x, network.add_constant((3,), order).get_output(0), 3).get_output(0)
    x = network.add_elementwise(x, network.add_constant(scale.shape, scale).get_output(0),
                                trt.ElementWiseOperation.PROD).get_output(0)
    x = network.add_elementwise(x, network.add_constant(shift.shape, shift).get_output(0),
                                trt.ElementWiseOperation.SUM).get_output(0)
    layout = network.add_shuffle(x)
    layout.first_transpose = (0, 3, 1, 2)
    x = layout.get_output(0)

    for i in range(network.num_layers):
        layer = network.get_layer(i)
        for j in range(layer.num_inputs):
            if layer.get_input(j) is not None and layer.get_input(j).name == model_input.name:
                layer.set_input(j, x)
    network.remove_tensor(model_input)
    return order, scale, shift


def prepare(path):
    """
    Build and cache a model's TensorRT engine on this computer — the
    one-time step for a new .pkl (minutes). Loading a model never builds:
    a Detection loads only a prepared one, in well under a second.

    Returns {"built": bool, "seconds": float}: built is False when the
    engine was already cached, or when the model has no ONNX to build
    (one still on OpenVINO).
    """
    data = _load_pickle(path)
    onnx_bytes = data.get("onnx")
    if onnx_bytes is None:
        return {"built": False, "seconds": 0.0}
    precision = _trt_precision(data)
    engine_path = _trt_engine_path(onnx_bytes, precision)
    started = time.time()
    with _TRT_BUILD_LOCK:
        if os.path.exists(engine_path):
            return {"built": False, "seconds": 0.0}
        builder = trt.Builder(_TRT_LOGGER)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, _TRT_LOGGER)
        if not parser.parse(onnx_bytes):
            errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise ValueError("Invalid ONNX model: %s" % errors)
        constants = _image_input(network)   # noqa: F841 — read by the build below
        config = builder.create_builder_config()
        if _TRT_PRECISIONS[precision] is not None:
            config.set_flag(_TRT_PRECISIONS[precision])
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, _TRT_WORKSPACE)
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise ValueError("TensorRT could not build an engine from this model.")
        os.makedirs(_TRT_CACHE, exist_ok=True)
        tmp = engine_path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(plan)
        os.replace(tmp, engine_path)
    return {"built": True, "seconds": round(time.time() - started, 1)}


def _trt_engine(onnx_bytes, precision):
    """
    The prepared TensorRT engine for an ONNX graph, from the cache.
    Returns (runtime, engine) — the runtime must outlive the engine.
    Raises ModelNotPreparedError when prepare() has not built it here.
    """
    path = _trt_engine_path(onnx_bytes, precision)
    if not os.path.exists(path):
        raise ModelNotPreparedError(
            "This model is not prepared on this unit: run model_prepare with its .pkl "
            "once (it builds the TensorRT engine for this GPU, a few minutes)."
        )
    with open(path, "rb") as f:
        plan = f.read()

    runtime = trt.Runtime(_TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise ValueError("Failed to load the TensorRT engine %s" % path)
    return runtime, engine


class _TRTModel:
    """A TensorRT engine that infers on one execution context PER THREAD.

    The vision server answers each websocket call on a thread-pool worker,
    so two detections against the same model overlap as a matter of
    course, and an execution context is not thread-safe. Each thread gets
    its own context — with its own device buffers and CUDA stream — made
    on first use; the engine (the weights) is shared. Inference stays
    concurrent: no lock, no queue. The pool's threads live as long as the
    server, so the contexts are freed together, in close().

    Built from a model pickle (its "onnx" and meta["precision"]). Call
    with the input arrays in engine order; returns {output name: array}.
    """

    def __init__(self, data):
        self._runtime, self._engine = _trt_engine(data["onnx"], _trt_precision(data))
        self.inputs, self.outputs = [], []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            tensor = (name, tuple(self._engine.get_tensor_shape(name)),
                      np.dtype(trt.nptype(self._engine.get_tensor_dtype(name))))
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(tensor)
            else:
                self.outputs.append(tensor)
        self._local = threading.local()
        self._slots = []
        self._slots_lock = threading.Lock()

    def _slot(self):
        slot = getattr(self._local, "slot", None)
        if slot is None:
            slot = SimpleNamespace(context=self._engine.create_execution_context(),
                                   stream=_cuda(cudart.cudaStreamCreate()),
                                   device={}, host={})
            for name, shape, dtype in self.inputs + self.outputs:
                slot.host[name] = np.empty(shape, dtype)
                slot.device[name] = _cuda(cudart.cudaMalloc(slot.host[name].nbytes))
                slot.context.set_tensor_address(name, slot.device[name])
            self._local.slot = slot
            with self._slots_lock:
                self._slots.append(slot)
        return slot

    def __call__(self, *arrays):
        slot = self._slot()
        for (name, _, dtype), array in zip(self.inputs, arrays):
            array = np.ascontiguousarray(array, dtype=dtype)
            _cuda(cudart.cudaMemcpyAsync(slot.device[name], array.ctypes.data, array.nbytes,
                                         cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, slot.stream))
        if not slot.context.execute_async_v3(slot.stream):
            raise RuntimeError("TensorRT inference failed")
        for name, _, _ in self.outputs:
            host = slot.host[name]
            _cuda(cudart.cudaMemcpyAsync(host.ctypes.data, slot.device[name], host.nbytes,
                                         cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, slot.stream))
        _cuda(cudart.cudaStreamSynchronize(slot.stream))
        return {name: slot.host[name].copy() for name, _, _ in self.outputs}

    def close(self):
        with self._slots_lock:
            for slot in self._slots:
                for pointer in slot.device.values():
                    cudart.cudaFree(pointer)
                cudart.cudaStreamDestroy(slot.stream)
                slot.context = None
            self._slots = []
        self._engine = None
        self._runtime = None


def _bbox_to_corners(x, y, w, h):
    return [
        [x,         y],
        [x + w,     y],
        [x + w,     y + h],
        [x,         y + h],
    ]


# ───────────────────────────────────────────────────────────────────────
# OD — Object detection (RF-DETR on TensorRT)
# ───────────────────────────────────────────────────────────────────────

class OD(object):
    """
    Axis-aligned object detector: RF-DETR, on the GPU with TensorRT.
    Single forward pass. DETR predicts a SET of objects — one query per
    object — so there is no NMS.

    Pickle:
        onnx   : bytes — RF-DETR's ONNX export: input "input" (1, 3, S, S);
                 outputs "dets" (1, Q, 4) cx, cy, w, h normalized to 0..1,
                 "labels" (1, Q, C) class logits
        cls    : list[str] — logit slot i is cls[i]; slots past it are unused
        colors : dict|list of hex strings
        meta   : {image_size: S, arch: "rf-detr", variant: str, precision: "fp16"}

    Call:
        OD(img, conf=0.3, cls=None, max_det=None) ->
        list[SimpleNamespace(prob, cls, rect, color)]

    `rect` is a SimpleNamespace(x, y, w, h) in original-image pixel coords.
    """

    def __init__(self, path, **kwargs):
        data = _load_pickle(path)
        if data.get("onnx") is None:
            raise ValueError(
                "Unsupported model: no 'onnx' entry. On the Jetson, OD is RF-DETR — "
                "train and export it with training_notebooks/train_object_detection.ipynb."
            )
        self.cls    = list(data["cls"])
        self.colors = data.get("colors", {})
        self.image_size = int(data["meta"]["image_size"])
        self.model = _TRTModel(data)

    def __del__(self):
        model = getattr(self, "model", None)
        if model is not None:
            model.close()
            self.model = None

    def _preprocess(self, img):
        # RF-DETR's own predict(): a plain (not letterboxed) bilinear resize
        # to S x S. RGB and ImageNet mean/std happen in the engine (_image_input).
        return cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)[None]

    def __call__(self, img, conf=0.3, cls=None, max_det=None, **kwargs):
        if img is None or img.size == 0:
            return []

        out = self.model(self._preprocess(img))
        boxes  = out["dets"][0]       # (Q, 4) cx, cy, w, h in 0..1
        logits = out["labels"][0]     # (Q, C)
        num_queries, num_slots = logits.shape

        # RF-DETR's selection: every (query, class) pair scored with a
        # sigmoid, best first, at most Q of them — one query may report
        # more than one class.
        scores = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
        pairs = np.flatnonzero(scores > conf)
        if len(pairs) == 0:
            return []
        pairs = pairs[np.argsort(-scores[pairs], kind="stable")][:num_queries]
        queries, class_ids = pairs // num_slots, pairs % num_slots

        H, W = img.shape[:2]
        cx, cy, w, h = boxes[queries].T
        x1 = ((cx - w * 0.5) * W).clip(0, W - 1)
        y1 = ((cy - h * 0.5) * H).clip(0, H - 1)
        x2 = ((cx + w * 0.5) * W).clip(0, W - 1)
        y2 = ((cy + h * 0.5) * H).clip(0, H - 1)

        objects = []
        # Three-state semantics:
        #   cls=None  → no filter, return every detection
        #   cls=[]    → empty filter, return nothing
        #   cls=[...] → only these labels pass
        cls_filter = None if cls is None else set(cls)
        for i, pair in enumerate(pairs):
            if class_ids[i] >= len(self.cls):
                continue
            label = self.cls[int(class_ids[i])]
            if cls_filter is not None and label not in cls_filter:
                continue
            objects.append(SimpleNamespace(
                prob=float(scores[pair]),
                cls=label,
                rect=SimpleNamespace(x=float(x1[i]), y=float(y1[i]),
                                     w=float(x2[i] - x1[i]), h=float(y2[i] - y1[i])),
                color=_color_for_class(class_ids[i], self.cls, self.colors),
            ))

        if max_det:
            objects = objects[:max_det]
        return objects


# ───────────────────────────────────────────────────────────────────────
# ROD — Rotated object detection (mmrotate / draft pickle format)
# ───────────────────────────────────────────────────────────────────────

class ROD(object):
    """
    Oriented (rotated) object detector. Per-class rotated NMS.

    Output objects expose:
        prob   : float
        cls    : str
        rect   : SimpleNamespace(cx, cy, w, h, angle_deg)
        corners: 4x[x,y] of the rotated rectangle (orig-image coords)
        color  : (B,G,R) tuple

    `angle_deg` is OpenCV-style (the angle returned by minAreaRect-friendly
    cv2.boxPoints input). Useful for grasp planning.
    """

    def __init__(self, path, device_name="CPU", device_config=None, **kwargs):
        data = _load_pickle(path)
        self.cls    = list(data["cls"])
        self.colors = data.get("colors", {})
        meta = data.get("meta", {})
        self.image_size = int(meta.get("image_size", 416))

        self.core, self.model, self.compiled_model = _read_openvino_from_pickle(
            data,
            input_shape=[1, 3, self.image_size, self.image_size],
            device_name=device_name,
            device_config=device_config,
        )

    def __del__(self):
        _release_openvino(getattr(self, "model", None), getattr(self, "compiled_model", None))
        self.core = None

    def _preprocess(self, img):
        padded, ratio = _letterbox(img, self.image_size)
        tensor = ((padded.astype(np.float32) - _DET_MEAN) / _DET_STD).transpose(2, 0, 1)[None]
        return tensor, ratio

    @staticmethod
    def _rotated_nms_class_agnostic(boxes_cxcywha, scores, class_ids, iou_thr):
        """
        Class-agnostic rotated NMS — one box per physical region. When the
        model emits two predictions for the same object with different
        classes (e.g. cardamom_good vs cardamom_bad on the same cardamom),
        the higher-confidence one wins and the other is suppressed. This
        is the right default for "one detection per object" use-cases like
        grasp planning. `class_ids` is unused; kept for signature parity
        with the per-class variant in OD.
        """
        if len(boxes_cxcywha) == 0:
            return np.array([], dtype=np.int64)
        rects = [((float(cx), float(cy)), (float(w), float(h)), float(ang))
                 for cx, cy, w, h, ang in boxes_cxcywha]
        picked = cv2.dnn.NMSBoxesRotated(rects, scores.tolist(),
                                         score_threshold=0.0,
                                         nms_threshold=float(iou_thr))
        if len(picked) == 0:
            return np.array([], dtype=np.int64)
        return np.array(picked).reshape(-1).astype(np.int64)

    def __call__(self, img, conf=0.15, nms_thr=0.3, cls=None, max_det=None, **kwargs):
        # `nms_iou` accepted as a kwarg alias for back-compat with newer call sites.
        nms_thr = float(kwargs.pop("nms_iou", nms_thr))
        if img is None or img.size == 0:
            return []

        tensor, ratio = self._preprocess(img)
        results = self.compiled_model([tensor])
        raw = np.asarray(list(results.values())[0])   # (1, N, 5+1+C)
        pred = raw[0]

        cx, cy, w, h, ang_rad = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3], pred[:, 4]
        obj = pred[:, 5]
        cls_scores = pred[:, 6:]

        class_ids = cls_scores.argmax(axis=1)
        max_cls   = cls_scores.max(axis=1)
        scores    = obj * max_cls

        keep_mask = scores > conf
        if not keep_mask.any():
            return []
        cx, cy, w, h, ang_rad = (cx[keep_mask], cy[keep_mask], w[keep_mask],
                                 h[keep_mask], ang_rad[keep_mask])
        scores, class_ids = scores[keep_mask], class_ids[keep_mask]

        # Un-letterbox center + size; angle is invariant under uniform scaling.
        cx = cx / ratio
        cy = cy / ratio
        w  = w  / ratio
        h  = h  / ratio

        H, W = img.shape[:2]
        cx = cx.clip(0, W - 1)
        cy = cy.clip(0, H - 1)

        ang_deg = np.degrees(ang_rad)
        boxes = np.stack([cx, cy, w, h, ang_deg], axis=1)

        keep_idx = self._rotated_nms_class_agnostic(boxes, scores, class_ids, nms_thr)
        if len(keep_idx) == 0:
            return []

        boxes, scores, class_ids = boxes[keep_idx], scores[keep_idx], class_ids[keep_idx]

        objects = []
        # Three-state semantics:
        #   cls=None  → no filter, return every detection
        #   cls=[]    → empty filter, return nothing
        #   cls=[...] → only these labels pass
        cls_filter = None if cls is None else set(cls)
        for i in range(len(boxes)):
            label = self.cls[int(class_ids[i])]
            if cls_filter is not None and label not in cls_filter:
                continue
            cx_i, cy_i, w_i, h_i, ang_i = boxes[i].tolist()
            corners = cv2.boxPoints(((cx_i, cy_i), (w_i, h_i), ang_i)).tolist()
            objects.append(SimpleNamespace(
                prob=float(scores[i]),
                cls=label,
                rect=SimpleNamespace(cx=float(cx_i), cy=float(cy_i),
                                     w=float(w_i), h=float(h_i),
                                     angle_deg=float(ang_i)),
                corners=[[float(p[0]), float(p[1])] for p in corners],
                color=_color_for_class(class_ids[i], self.cls, self.colors),
            ))

        objects.sort(key=lambda o: o.prob, reverse=True)
        if max_det:
            objects = objects[:max_det]
        return objects


# ───────────────────────────────────────────────────────────────────────
# KP — Top-down keypoint detection (mmpose / draft pickle format)
# ───────────────────────────────────────────────────────────────────────

class KP(object):
    """
    Top-down keypoint detector. Runs on a SINGLE object's region.

    Two equivalent ways to use it (your choice):

      1. Pass a pre-cropped image (you handle the crop yourself):
             kp(crop)
      2. Pass a full image plus a bbox (we apply the same affine the
         training pipeline used — proportional fit + 1.25x padding):
             kp(image, bbox=(x, y, w, h))

    Compose with OD/ROD:
        objs = od(image)
        for obj in objs:
            keypoints = kp(image, bbox=(obj.rect.x, obj.rect.y,
                                        obj.rect.w, obj.rect.h))

    Returns SimpleNamespace(
        keypoints = [SimpleNamespace(name, x, y, conf, color), ...],
        skeleton  = [(i, j), ...]   # 1-indexed per mmpose convention
    )
    """

    def __init__(self, path, device_name="CPU", device_config=None, padding=1.25, **kwargs):
        data = _load_pickle(path)
        self.keypoint_names = list(data.get("keypoint_names", []))
        self.skeleton       = list(data.get("skeleton", []))
        self.colors         = data.get("colors", {})
        meta = data.get("meta", {})
        self.input_w = int(meta.get("input_w", 192))
        self.input_h = int(meta.get("input_h", 256))
        self.padding = float(padding)

        self.core, self.model, self.compiled_model = _read_openvino_from_pickle(
            data,
            input_shape=[1, 3, self.input_h, self.input_w],
            device_name=device_name,
            device_config=device_config,
        )

    def __del__(self):
        _release_openvino(getattr(self, "model", None), getattr(self, "compiled_model", None))
        self.core = None

    @staticmethod
    def _third_point(a, b):
        d = a - b
        return b + np.array([-d[1], d[0]], dtype=np.float32)

    def _affine_warp(self, img, bbox, padding=None):
        """
        Build the 3-point affine that crops `bbox` out of `img` while
        preserving aspect ratio + applying the training-time padding.
        Returns (warped_input, warp_2x3).

        Padding precedence:
          1. explicit `padding` kwarg (highest)
          2. bbox is None → 1.0 (caller passed a pre-cropped image, so
             adding more padding would double-pad it)
          3. bbox given   → self.padding (1.25 by default — matches the
             training-time top-down framing)
        """
        if padding is not None:
            pad = float(padding)
        elif bbox is None:
            pad = 1.0
        else:
            pad = self.padding
        H, W = img.shape[:2]
        if bbox is None:
            x, y, w, h = 0, 0, W, H
        else:
            x, y, w, h = bbox
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        ar = self.input_w / self.input_h
        bw, bh = float(w), float(h)
        if bw > ar * bh:
            bh = bw / ar
        else:
            bw = bh * ar
        scale_w = bw * pad
        scale_h = bh * pad   # noqa: F841  (kept for symmetry / future use)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src_dir = np.array([0, scale_w * -0.5], dtype=np.float32)
        dst_dir = np.array([0, self.input_w * -0.5], dtype=np.float32)
        src[0] = center
        src[1] = center + src_dir
        src[2] = self._third_point(src[0], src[1])
        dst[0] = [self.input_w * 0.5, self.input_h * 0.5]
        dst[1] = dst[0] + dst_dir
        dst[2] = self._third_point(dst[0], dst[1])

        warp = cv2.getAffineTransform(src, dst)
        crop = cv2.warpAffine(img, warp, (self.input_w, self.input_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(_PAD_VALUE, _PAD_VALUE, _PAD_VALUE))
        return crop, warp

    def _preprocess(self, crop_bgr):
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        return ((rgb - _IMNET_MEAN) / _IMNET_STD).transpose(2, 0, 1)[None]

    def _kp_color(self, idx):
        # Fall back to a stable palette if no per-keypoint colour was stored.
        palette = [(51, 153, 255), (0, 255, 0), (255, 128, 0), (255, 51, 255),
                   (0, 255, 255), (255, 255, 0), (128, 0, 255), (255, 0, 128)]
        if isinstance(self.colors, dict) and self.colors:
            name = self.keypoint_names[idx] if idx < len(self.keypoint_names) else None
            if name and name in self.colors:
                return _hex_to_bgr(self.colors[name])
        if isinstance(self.colors, (list, tuple)) and len(self.colors) > 0:
            return _hex_to_bgr(self.colors[idx % len(self.colors)])
        return palette[idx % len(palette)]

    def __call__(self, img, bbox=None, conf=0.3, padding=None, **kwargs):
        if img is None or img.size == 0:
            return SimpleNamespace(keypoints=[], skeleton=list(self.skeleton))

        crop, warp = self._affine_warp(img, bbox, padding=padding)
        tensor = self._preprocess(crop)
        results = self.compiled_model([tensor])
        kp_out = np.asarray(list(results.values())[0])[0]   # (K, 3) in input coords

        # Map back to original image coordinates.
        inv = cv2.invertAffineTransform(warp)
        xy = kp_out[:, :2]
        xy_h = np.hstack([xy, np.ones((xy.shape[0], 1), dtype=np.float32)])
        xy_orig = xy_h @ inv.T
        confs = kp_out[:, 2]

        keypoints = []
        for i in range(xy_orig.shape[0]):
            c = float(confs[i])
            if c < conf:
                continue
            name = self.keypoint_names[i] if i < len(self.keypoint_names) else f"kp{i}"
            keypoints.append(SimpleNamespace(
                name=name,
                x=float(xy_orig[i, 0]),
                y=float(xy_orig[i, 1]),
                conf=c,
                color=self._kp_color(i),
            ))

        return SimpleNamespace(keypoints=keypoints, skeleton=list(self.skeleton))


# ───────────────────────────────────────────────────────────────────────
# CLS — Classification (cls / draft pickle format)
# ───────────────────────────────────────────────────────────────────────

class CLS(object):
    """
    Image classifier (softmax over named classes): DINOv3 ViT-S/16 features
    and a linear head, one graph, on the GPU with TensorRT.

    Pickle:
        onnx   : bytes — input "input" (1, 3, H, W): the centred letterbox,
                 RGB, ImageNet mean/std; output "logits" (1, C)
        cls    : list[str]
        colors : dict|list of hex strings
        meta   : {image_size: (W, H), type: "cls", arch: "dinov3-linear", model,
                  precision: "bf16"}

    Call:
        CLS(img, conf=0.5, top_k=1) -> list[[name, prob, color], ...]
        # legacy shape: a list of 3-element lists, sorted by prob desc.

    Empty list when no class clears `conf`. `top_k` caps the result.
    """

    def __init__(self, path, **kwargs):
        data = _load_pickle(path)
        if data.get("onnx") is None:
            raise ValueError(
                "Unsupported model: no 'onnx' entry. On the Jetson, CLS is DINOv3 — "
                "train and export it with training_notebooks/train_classification.ipynb."
            )
        self.cls    = list(data["cls"])
        self.colors = data.get("colors", {})
        self.image_size = tuple(data["meta"]["image_size"])   # (W, H)
        self.model = _TRTModel(data)

    def __del__(self):
        model = getattr(self, "model", None)
        if model is not None:
            model.close()
            self.model = None

    def _preprocess(self, img):
        # RGB and ImageNet mean/std happen in the engine (_image_input)
        return _letterbox_centered(img, self.image_size)[None]

    @staticmethod
    def _softmax(x):
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def __call__(self, img, conf=0.5, top_k=1, **kwargs):
        if img is None or img.size == 0:
            return []
        logits = self.model(self._preprocess(img))["logits"].reshape(-1)
        probs  = self._softmax(logits)

        order = np.argsort(probs)[::-1]
        out = []
        for idx in order[:max(int(top_k), 1)]:
            p = float(probs[idx])
            if p < conf:
                break
            label = self.cls[int(idx)]
            out.append([label, p, _color_for_class(idx, self.cls, self.colors)])
        return out


# ───────────────────────────────────────────────────────────────────────
# ANOM — Anomaly detection (anomaly/stfpm / draft pickle format)
# ───────────────────────────────────────────────────────────────────────

class ANOM(object):
    """
    STFPM-style anomaly detection — single pass/fail verdict per image.

    Pickle:
        xml, bin
        cls    : ["good", "fail"]                # or similar two-class
        colors : dict|list of hex strings
        meta   : {image_size: (W, H), type: "anom", clahe: {...}|None,
                  threshold: float}

    Call:
        ANOM(img, threshold=None) -> SimpleNamespace(
            cls, prob, score, threshold, color
        )

    `cls`       — first class ("good") if score ≤ threshold, else second ("fail").
    `score`     — image-level anomaly score (max of the model's heat-map).
    `prob`      — alias for `score`, lets ANOM slot into the same downstream
                  prob/cls/color code paths as OD/CLS.
    `threshold` — the threshold actually used (passed-in or pickle-embedded).
    `color`     — BGR colour for the chosen class.

    Heat-map is intentionally NOT returned. Anomaly detection is a one-class
    problem — one verdict per part. Per-pixel localization is diagnostic
    information for offline review, not something to surface as separate
    "detections".
    """

    def __init__(self, path, device_name="CPU", device_config=None, **kwargs):
        data = _load_pickle(path)
        self.cls    = list(data["cls"])
        self.colors = data.get("colors", {})
        meta = data.get("meta", {})
        self.image_size = tuple(meta.get("image_size", (256, 256)))   # (W, H)
        self.clahe_cfg  = meta.get("clahe") or None
        self.threshold  = meta.get("threshold")

        self.core, self.model, self.compiled_model = _read_openvino_from_pickle(
            data,
            input_shape=[1, 3, self.image_size[1], self.image_size[0]],
            device_name=device_name,
            device_config=device_config,
        )

    def __del__(self):
        _release_openvino(getattr(self, "model", None), getattr(self, "compiled_model", None))
        self.core = None

    def _apply_clahe(self, image_bgr):
        if not self.clahe_cfg:
            return image_bgr
        clip_limit = self.clahe_cfg.get("clip_limit", 2.0)
        tile_size  = tuple(self.clahe_cfg.get("tile_size", (8, 8)))
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l_eq = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

    def _preprocess(self, img):
        img = self._apply_clahe(img)
        W, H = self.image_size
        resized = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        return ((rgb - _IMNET_MEAN) / _IMNET_STD).transpose(2, 0, 1)[None]

    def __call__(self, img, threshold=None, **kwargs):
        if img is None or img.size == 0:
            return SimpleNamespace(cls=None, prob=0.0, score=0.0,
                                   threshold=None, color=(255, 255, 255))

        tensor = self._preprocess(img)
        results = self.compiled_model([tensor])

        # The model returns either a single scalar (image-level score) or
        # a (scalar, heat-map) pair. We take the scalar; the heat-map is
        # ignored — anomaly is a one-class problem and we report a single
        # verdict per image, not per-region detections.
        score = None
        heatmap_max = None
        for v in results.values():
            arr = np.asarray(v)
            if arr.size == 1:
                score = float(arr.squeeze())
            else:
                # If the model only emits a heat-map (no scalar), use its
                # max as the image-level score (standard convention).
                heatmap_max = float(arr.max())
        if score is None:
            score = heatmap_max if heatmap_max is not None else 0.0

        thr = self.threshold if threshold is None else float(threshold)

        # Verdict — first class is the "pass" label (e.g. "good"), second
        # is the "fail" label. score > threshold → fail.
        cls_label = None
        cls_idx = None
        if thr is not None and len(self.cls) >= 2:
            cls_idx = 1 if score > thr else 0
            cls_label = self.cls[cls_idx]

        color = _color_for_class(cls_idx, self.cls, self.colors) if cls_idx is not None else (255, 255, 255)

        return SimpleNamespace(
            cls=cls_label,
            prob=score,           # alias for `.score`, slots into OD/CLS-shaped consumers
            score=score,
            threshold=thr,
            color=color,
        )


def letterbox_image(image, target_size=(640, 640), pad_color=(128, 128, 128)):
    """Centered letterbox used by OCR."""
    ih, iw = image.shape[:2]
    tw, th = target_size
    scale = min(tw / iw, th / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    image_resized = cv2.resize(image, (nw, nh))

    pad_w = (tw - nw) // 2
    pad_h = (th - nh) // 2

    top = pad_h
    bottom = th - nh - pad_h
    left = pad_w
    right = tw - nw - pad_w

    new_image = cv2.copyMakeBorder(image_resized, top, bottom, left, right,
                                   borderType=cv2.BORDER_CONSTANT, value=pad_color)
    return new_image, scale, pad_w, pad_h


class OCR(object):
    """OpenVINO horizontal-text detection + recognition pipeline. Unchanged."""

    def __init__(self, device="CPU"):
        spec = importlib.util.find_spec("dorna_vision")
        if spec and spec.origin:
            model_folder = os.path.dirname(spec.origin)

        det_model_path = os.path.join(model_folder, "model", "ocr", "horizontal-text-detection-0001.xml")
        det_weights_path = os.path.join(model_folder, "model", "ocr", "horizontal-text-detection-0001.bin")
        rec_model_path = os.path.join(model_folder, "model", "ocr", "inference.pdmodel")
        dict_path = os.path.join(model_folder, "model", "ocr", "ppocr_keys_v1.txt")

        if not os.path.exists(dict_path):
            raise FileNotFoundError(f"Dictionary file {dict_path} not found.")
        with open(dict_path, "r", encoding="utf-8") as f:
            keys = [line.strip() for line in f if line.strip() != ""]
        self.char_list = [""] + keys

        from openvino import Core   # OCR is not yet on TensorRT
        self.core = Core()
        if not os.path.exists(det_model_path) or not os.path.exists(det_weights_path):
            raise FileNotFoundError("Detection model files not found.")
        self.det_model = self.core.read_model(model=det_model_path, weights=det_weights_path)
        self.det_model.reshape({self.det_model.input(0): [1, 3, 640, 640]})
        self.det_compiled = self.core.compile_model(self.det_model, device)
        self.det_input = self.det_compiled.input(0)
        self.det_output = self.det_compiled.output(0)

        if not os.path.exists(rec_model_path):
            raise FileNotFoundError("Recognition model file not found.")
        self.rec_model = self.core.read_model(model=rec_model_path)
        self.rec_model.reshape({self.rec_model.input(0): [1, 3, 48, -1]})
        self.rec_compiled = self.core.compile_model(self.rec_model, device)
        self.rec_input = self.rec_compiled.input(0)
        self.rec_output = self.rec_compiled.output(0)

    def __del__(self):
        try:
            del self.det_model
            del self.det_compiled
            del self.rec_model
            del self.rec_compiled
            self.core = None
        except Exception:
            pass

    def preprocess_crop(self, crop, target_height=48, target_width=320):
        h, w, _ = crop.shape
        ratio = w / h
        new_w = int(math.ceil(target_height * ratio))
        new_w = min(new_w, target_width)
        resized = cv2.resize(crop, (new_w, target_height))
        resized = resized.astype("float32") / 255.0
        resized = (resized - 0.5) / 0.5
        resized = resized.transpose(2, 0, 1)
        padded = np.zeros((3, target_height, target_width), dtype="float32")
        padded[:, :, :new_w] = resized
        return np.expand_dims(padded, 0)

    def ctc_decode(self, logits, char_list):
        indices = np.argmax(logits, axis=2)[0]
        decoded = []
        prev = -1
        for idx in indices:
            if idx != prev and idx != 0:
                decoded.append(char_list[idx])
            prev = idx
        return "".join(decoded)

    def ocr(self, img, conf=0.5, detection_enabled=True, **kwargs):
        results = []
        if detection_enabled:
            det_img, scale, pad_w, pad_h = letterbox_image(img, (640, 640))
            det_input_img = det_img.astype("float32")
            det_input_img = det_input_img.transpose(2, 0, 1)
            det_input_img = np.expand_dims(det_input_img, 0)

            det_result = self.det_compiled([det_input_img])[self.det_output]
            det_result = np.array(det_result)
            if det_result.ndim == 3:
                det_result = np.squeeze(det_result, axis=0)

            for b in det_result:
                x1, y1, x2, y2, score = b
                if score < conf:
                    continue
                x_min_letter = min(x1, x2); x_max_letter = max(x1, x2)
                y_min_letter = min(y1, y2); y_max_letter = max(y1, y2)
                x_min = int((x_min_letter - pad_w) / scale)
                x_max = int((x_max_letter - pad_w) / scale)
                y_min = int((y_min_letter - pad_h) / scale)
                y_max = int((y_max_letter - pad_h) / scale)
                if x_min >= x_max or y_min >= y_max:
                    continue
                x_min = max(0, x_min); y_min = max(0, y_min)
                x_max = min(img.shape[1], x_max); y_max = min(img.shape[0], y_max)
                crop = img[y_min:y_max, x_min:x_max]
                if crop.size == 0:
                    continue
                rec_input_crop = self.preprocess_crop(crop)
                rec_result = self.rec_compiled([rec_input_crop])[self.rec_output]
                text = self.ctc_decode(rec_result, self.char_list)
                results.append([
                    [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]],
                    [text, float(score)],
                ])
        else:
            rec_input_img = self.preprocess_crop(img)
            rec_result = self.rec_compiled([rec_input_img])[self.rec_output]
            text = self.ctc_decode(rec_result, self.char_list)
            h, w = img.shape[:2]
            results.append([
                [[0, 0], [0, h], [w, h], [w, 0]],
                [text, 1],
            ])
        return results
