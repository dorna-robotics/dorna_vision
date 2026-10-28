"""
ML inference primitives for the dorna_vision pipeline (Jetson).

Each class is a single-purpose, init-once / call-many wrapper around a
model packaged in a pickle, run on the GPU with TensorRT. The pickle
carries a portable ONNX graph and how to build and feed it; prepare()
builds the TensorRT engine on this computer, once per model (an engine is
tied to the GPU and the TensorRT version that built it), and caches it.
Loading a model never builds — an unprepared one raises
ModelNotPreparedError. The pickle schema (written by the training
notebooks under training_notebooks/):

    {
        "onnx":   <ONNX graph, bytes>    # one float input (1, 3, H, W)
        "cls":    ["class_a", ...]       # the model's class slots, in order
        "colors": {"class_a": "#aabbcc", ...}   # dict OR list of hex
        "meta": {
            "type":       "od" | "seg" | "kp" | "cls",
            "image_size": S (od/seg/kp) | (W, H) (cls),
            "precision":  "fp16" | "bf16" | "fp32",   # the engine's
            "input":      {"format": "uint8-bgr" | "float", ...},  # what the
                                                       # engine takes (_engine_spec)
            "profile":    {"min", "opt", "max"},      # variable-shape graphs only
            ...                                       # see each class
        },
    }

    OD   — RF-DETR: boxes
    SEG  — RF-DETR Seg: boxes + outlines
    KP   — RF-DETR keypoints: boxes + each object's keypoints
    CLS  — DINOv3 ViT-S/16 + a linear head: one label per image
    OCR  — PP-OCRv6: text lines and their text (built-in models, fetch_ocr())
"""

import hashlib
import json
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


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ───────────────────────────────────────────────────────────────────────
# TensorRT — the Jetson GPU
# ───────────────────────────────────────────────────────────────────────

# Engines built from a pickle's ONNX, cached by everything the build
# depends on — the ONNX, its precision and input, the TensorRT version,
# the GPU, the builder's workspace: an engine runs only on the GPU and the
# TensorRT that built it, so it is built here, never shipped.
_TRT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "dorna_vision", "trt")

# The builder's scratch memory while it times candidate kernels. The GPU
# shares the Jetson's 8 GB with everything else; unbounded, the builder
# grabs what happens to be free and skips the kernels that did not fit —
# the engine would depend on what else was running. Fixed, it does not.
_TRT_WORKSPACE = 1 << 30

# One build at a time: a build holds the GPU for minutes.
_TRT_BUILD_LOCK = threading.Lock()

_TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# The precisions a model may declare (meta["precision"]), as the builder
# flag each sets; "fp32" sets none. The training notebook picks it for
# its architecture: FP16 where the activations fit its range (RF-DETR),
# BF16 — FP32's range, half the bits, as fast on Orin — where they do
# not: DINOv3's attention scores (Q·Kᵀ, queries and keys in the hundreds)
# overflow FP16's 65504 and every output turns NaN.
_TRT_PRECISIONS = {"fp16": trt.BuilderFlag.FP16, "bf16": trt.BuilderFlag.BF16, "fp32": None}


def _cuda(result):
    """Unwrap a cuda-python call: raise on an error, return the value(s)."""
    err, *rest = result
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError("CUDA: %s" % cudart.cudaGetErrorString(err)[1].decode())
    return rest[0] if len(rest) == 1 else None


class ModelNotPreparedError(ValueError):
    """The model's TensorRT engine is not built on this computer yet."""


def _engine_spec(data):
    """
    What a model's engine is built from, validated — a pickle missing any
    of it is refused with what to do, never guessed:

        onnx       the graph: one float input (N, 3, H, W)
        precision  meta["precision"]
        input      meta["input"]: what the ENGINE takes —
                     {"format": "uint8-bgr", "channels": "rgb"|"bgr",
                      "mean": [3], "std": [3]}  the camera image, (N, H, W, 3)
                      uint8 BGR, normalised on the GPU into the graph's input
                      (_image_input); channels / mean / std (0..1) are the
                      graph's;
                     {"format": "float"}  the graph's own float input, as the
                      caller computes it
        profile    meta["profile"]: {"min", "opt", "max"} — the graph input's
                     (N, 3, H, W) range when the graph has variable dims;
                     None when it has none
    """
    meta = data.get("meta") or {}
    onnx_bytes = data.get("onnx")
    precision = meta.get("precision")
    model_input = dict(meta.get("input") or {})
    profile = meta.get("profile")
    if onnx_bytes is None:
        raise ValueError("Invalid model: no 'onnx' entry — export it with its training notebook.")
    if precision not in _TRT_PRECISIONS:
        raise ValueError("Invalid model: meta['precision'] must be one of %s (got %r) — re-export it "
                         "from its training notebook." % (sorted(_TRT_PRECISIONS), precision))
    if model_input.get("format") == "uint8-bgr":
        if (model_input.get("channels") not in ("rgb", "bgr")
                or len(model_input.get("mean") or ()) != 3 or len(model_input.get("std") or ()) != 3):
            raise ValueError("Invalid model: a uint8-bgr input needs channels 'rgb'|'bgr', mean [3], std [3] "
                             "(got %r) — re-export it from its training notebook." % (model_input,))
        model_input = {"format": "uint8-bgr", "channels": model_input["channels"],
                       "mean": [float(v) for v in model_input["mean"]],
                       "std": [float(v) for v in model_input["std"]]}
    elif model_input.get("format") == "float":
        model_input = {"format": "float"}
    else:
        raise ValueError("Invalid model: meta['input']['format'] must be 'uint8-bgr' or 'float' (got %r) — "
                         "re-export it from its training notebook." % (model_input,))
    if profile is not None:
        profile = {k: [int(v) for v in profile[k]] for k in ("min", "opt", "max")}
    return onnx_bytes, precision, model_input, profile


def _engine_path(data):
    """Where this model's engine lives in the cache."""
    onnx_bytes, precision, model_input, profile = _engine_spec(data)
    prop = _cuda(cudart.cudaGetDeviceProperties(0))
    key = hashlib.sha256()
    key.update(onnx_bytes)
    key.update(json.dumps({"trt": trt.__version__, "gpu": prop.name.decode(), "sm": [prop.major, prop.minor],
                           "precision": precision, "input": model_input, "profile": profile,
                           "workspace": _TRT_WORKSPACE, "image_input": "onnx"}, sort_keys=True).encode())
    return os.path.join(_TRT_CACHE, key.hexdigest()[:16] + ".engine")


def _graph_input(onnx_bytes):
    """The graph's one input: (onnx model, its input, variable dims?) —
    refused unless it is one float (N, 3, H, W)."""
    import onnx
    model = onnx.load_from_string(onnx_bytes)
    inputs = [i for i in model.graph.input if i.name not in {t.name for t in model.graph.initializer}]
    if len(inputs) != 1:
        raise ValueError("Invalid ONNX model: one input expected, got %d" % len(inputs))
    graph_input = inputs[0]
    dims = graph_input.type.tensor_type.shape.dim
    if graph_input.type.tensor_type.elem_type != onnx.TensorProto.FLOAT or len(dims) != 4 or dims[1].dim_value != 3:
        raise ValueError("Invalid ONNX model: one float input (N, 3, H, W) expected, got %s" % graph_input)
    return model, graph_input, any(not d.HasField("dim_value") for d in dims)


def _image_input(model, graph_input, model_input):
    """
    Give the graph the camera's image as its input: uint8, (N, H, W, 3),
    BGR — what OpenCV hands us. Nodes prepended to the ONNX compute the
    graph's own float (N, 3, H, W) input from it on the GPU, as the model
    declares (meta["input"]): cast, BGR→RGB when it wants RGB,
    x·(1/255σ) − μ/σ, NHWC→NCHW. The CPU only resizes / letterboxes
    (OpenCV, uint8), and a quarter of the bytes cross to the GPU. Done in
    the ONNX, before TensorRT parses it, so TensorRT sees one graph with
    the input's own named dimensions — variable shapes included.
    """
    from onnx import helper, numpy_helper, TensorProto
    graph = model.graph
    n, _, h, w = graph_input.type.tensor_type.shape.dim
    image = helper.make_tensor_value_info("image", TensorProto.UINT8, None)
    for d in (n, h, w):
        image.type.tensor_type.shape.dim.add().CopyFrom(d)
    image.type.tensor_type.shape.dim.add().dim_value = 3

    mean = np.array(model_input["mean"], dtype=np.float32)
    std = np.array(model_input["std"], dtype=np.float32)
    constants = {"image_scale": (1.0 / (255.0 * std)).reshape(1, 1, 1, 3),
                 "image_shift": (-mean / std).reshape(1, 1, 1, 3),
                 "image_rgb": np.array([2, 1, 0], dtype=np.int64)}
    graph.initializer.extend(numpy_helper.from_array(v, name=k) for k, v in constants.items())
    nodes = [helper.make_node("Cast", ["image"], ["image_float"], to=TensorProto.FLOAT)]
    x = "image_float"
    if model_input["channels"] == "rgb":
        nodes.append(helper.make_node("Gather", [x, "image_rgb"], ["image_channels"], axis=3))
        x = "image_channels"
    nodes += [helper.make_node("Mul", [x, "image_scale"], ["image_scaled"]),
              helper.make_node("Add", ["image_scaled", "image_shift"], ["image_normalised"]),
              helper.make_node("Transpose", ["image_normalised"], [graph_input.name], perm=[0, 3, 1, 2])]
    graph.input.remove(graph_input)
    graph.input.insert(0, image)
    for i, node in enumerate(nodes):
        graph.node.insert(i, node)
    return model.SerializeToString()


def prepare(path):
    """
    Build and cache a model's TensorRT engine on this computer — the
    one-time step for a new .pkl (minutes). Loading a model never builds:
    a Detection loads only a prepared one, in well under a second.

    Returns {"built": bool, "seconds": float}: built is False when the
    engine was already cached.
    """
    data = _load_pickle(path)
    onnx_bytes, precision, model_input, profile = _engine_spec(data)
    engine_path = _engine_path(data)
    started = time.time()
    with _TRT_BUILD_LOCK:
        if os.path.exists(engine_path):
            return {"built": False, "seconds": 0.0}
        model, graph_input, variable = _graph_input(onnx_bytes)
        if variable != (profile is not None):
            raise ValueError("Invalid model: the graph input %s, meta['profile'] %s — a profile goes with "
                             "variable dims, and only with them." % (
                                 "has variable dims" if variable else "is fixed", "given" if profile else "missing"))
        if model_input["format"] == "uint8-bgr":
            onnx_bytes = _image_input(model, graph_input, model_input)
        builder = trt.Builder(_TRT_LOGGER)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, _TRT_LOGGER)
        if not parser.parse(onnx_bytes):
            errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise ValueError("Invalid ONNX model: %s" % errors)
        config = builder.create_builder_config()
        if profile is not None:
            nhwc = model_input["format"] == "uint8-bgr"
            shape = (lambda s: (s[0], s[2], s[3], s[1])) if nhwc else tuple
            optimization = builder.create_optimization_profile()
            optimization.set_shape(network.get_input(0).name, shape(profile["min"]), shape(profile["opt"]),
                                   shape(profile["max"]))
            config.add_optimization_profile(optimization)
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


class _TRTModel:
    """A TensorRT engine that infers on one execution context PER THREAD.

    The vision server answers each websocket call on a thread-pool worker,
    so two detections against the same model overlap as a matter of
    course, and an execution context is not thread-safe. Each thread gets
    its own context — with its own device buffers and CUDA stream — made
    on first use; the engine (the weights) is shared. Inference stays
    concurrent: no lock, no queue. The pool's threads live as long as the
    server, so the contexts are freed together, in close().

    An engine with a profile takes any input shape within it: each call
    sets its shape, and a thread's device buffers grow to the largest
    input it has seen — never sized to the profile's maximum up front.

    Built from a model pickle; raises ModelNotPreparedError when prepare()
    has not built its engine here. Call with the engine's input array (the
    uint8 BGR image, or the graph's float input — meta["input"]); returns
    {output name: array}.
    """

    def __init__(self, data):
        path = _engine_path(data)
        if not os.path.exists(path):
            raise ModelNotPreparedError(
                "This model is not prepared on this unit: run model_prepare with its .pkl "
                "once (it builds the TensorRT engine for this GPU, a few minutes)."
            )
        with open(path, "rb") as f:
            plan = f.read()
        self._runtime = trt.Runtime(_TRT_LOGGER)
        self._engine = self._runtime.deserialize_cuda_engine(plan)
        if self._engine is None:
            raise ValueError("Failed to load the TensorRT engine %s" % path)
        self.inputs, self.outputs = [], []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            tensor = (name, np.dtype(trt.nptype(self._engine.get_tensor_dtype(name))))
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
                                   device={}, capacity={})
            self._local.slot = slot
            with self._slots_lock:
                self._slots.append(slot)
        return slot

    @staticmethod
    def _buffer(slot, name, nbytes):
        """The slot's device buffer for a tensor, grown when it is too small."""
        if slot.capacity.get(name, -1) < nbytes:
            if name in slot.device:
                _cuda(cudart.cudaFree(slot.device[name]))
            slot.device[name] = _cuda(cudart.cudaMalloc(max(nbytes, 1)))
            slot.capacity[name] = nbytes
        slot.context.set_tensor_address(name, slot.device[name])
        return slot.device[name]

    def __call__(self, *arrays):
        slot = self._slot()
        for (name, dtype), array in zip(self.inputs, arrays):
            array = np.ascontiguousarray(array, dtype=dtype)
            if not slot.context.set_input_shape(name, array.shape):
                raise ValueError("Input %s of shape %s is outside the model's profile" % (name, array.shape))
            _cuda(cudart.cudaMemcpyAsync(self._buffer(slot, name, array.nbytes), array.ctypes.data, array.nbytes,
                                         cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, slot.stream))
        results = {}
        for name, dtype in self.outputs:
            results[name] = np.empty(tuple(slot.context.get_tensor_shape(name)), dtype)
            self._buffer(slot, name, results[name].nbytes)
        if not slot.context.execute_async_v3(slot.stream):
            raise RuntimeError("TensorRT inference failed")
        for name, host in results.items():
            _cuda(cudart.cudaMemcpyAsync(host.ctypes.data, slot.device[name], host.nbytes,
                                         cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, slot.stream))
        _cuda(cudart.cudaStreamSynchronize(slot.stream))
        return results

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


# ───────────────────────────────────────────────────────────────────────
# RF-DETR — OD, SEG, KP
# ───────────────────────────────────────────────────────────────────────

class _RFDETR(object):
    """
    The half OD, SEG and KP share: an RF-DETR model on TensorRT. RF-DETR's
    own predict(): a plain (not letterboxed) bilinear resize to S x S, one
    forward pass. DETR predicts a SET of objects — one query per object —
    so there is no NMS: every (query, class) pair is scored with a
    sigmoid, best first, at most Q of them (one query may report more than
    one class).

    Pickle: onnx; cls (logit slot i is cls[i]; None, or a slot past the
    list, is one the model does not use — a pretrained model's sparse COCO
    ids, a keypoint model's background slot); colors; meta {type,
    image_size: S, arch: "rf-detr", variant, precision, input}. The ONNX outputs "dets" (1, Q, 4) cx, cy, w, h
    normalised to 0..1 and "labels" (1, Q, C) class logits, plus each
    head's own (SEG, KP).
    """

    def __init__(self, path, **kwargs):
        data = _load_pickle(path)
        self.cls    = list(data["cls"])
        self.colors = data.get("colors", {})
        self.image_size = int(data["meta"]["image_size"])
        # which logit slots the model names; one more False for any slot past the list
        self._named = np.array([name is not None for name in self.cls] + [False])
        self.model = _TRTModel(data)
        self._load(data)

    def _load(self, data):
        """A head's own pickle entries."""

    def __del__(self):
        model = getattr(self, "model", None)
        if model is not None:
            model.close()
            self.model = None

    def _preprocess(self, img):
        # The resize of RF-DETR's training (Albumentations: OpenCV bilinear
        # on the uint8 image); RGB and the mean/std happen in the engine
        # (_image_input). (RF-DETR's predict() resizes the float image
        # instead — an approximation of training that moves borderline
        # scores by a few hundredths.)
        return cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)[None]

    @staticmethod
    def _ranked(logits):
        """The Q best (query, class) pairs, best first — RF-DETR's top-k:
        (queries, class ids, scores). The top Q are picked in linear time,
        then only they are sorted."""
        num_queries, num_slots = logits.shape
        scores = _sigmoid(logits.reshape(-1))
        top = np.argpartition(-scores, num_queries - 1)[:num_queries]
        top = top[np.argsort(-scores[top], kind="stable")]
        return top // num_slots, top % num_slots, scores[top]

    @staticmethod
    def _boxes(dets, queries, W, H):
        """The queries' boxes in image pixels, (x1, y1, x2, y2) each clipped to the image."""
        cx, cy, w, h = dets[queries].T
        return (((cx - w * 0.5) * W).clip(0, W - 1), ((cy - h * 0.5) * H).clip(0, H - 1),
                ((cx + w * 0.5) * W).clip(0, W - 1), ((cy + h * 0.5) * H).clip(0, H - 1))

    def _kept(self, class_ids, scores, conf, cls):
        """The ranked pairs to report: above conf, a slot the model names,
        through the cls filter. Three-state cls: None → every class,
        [] → none, [...] → only these."""
        keep = np.flatnonzero((scores > conf) & self._named[np.minimum(class_ids, len(self.cls))])
        if cls is not None:
            wanted = set(cls)
            keep = [i for i in keep if self.cls[class_ids[i]] in wanted]
        return list(keep)

    def _object(self, class_id, score, x1, y1, x2, y2):
        return SimpleNamespace(
            prob=float(score),
            cls=self.cls[int(class_id)],
            rect=SimpleNamespace(x=float(x1), y=float(y1), w=float(x2 - x1), h=float(y2 - y1)),
            color=_color_for_class(class_id, self.cls, self.colors),
        )


class OD(_RFDETR):
    """
    Axis-aligned object detector: RF-DETR.

    Call:
        OD(img, conf=0.3, cls=None, max_det=None) ->
        list[SimpleNamespace(prob, cls, rect, color)]

    `rect` is a SimpleNamespace(x, y, w, h) in original-image pixel coords.
    """

    def __call__(self, img, conf=0.3, cls=None, max_det=None, **kwargs):
        if img is None or img.size == 0:
            return []
        out = self.model(self._preprocess(img))
        queries, class_ids, scores = self._ranked(out["labels"][0])
        x1, y1, x2, y2 = self._boxes(out["dets"][0], queries, img.shape[1], img.shape[0])
        objects = [self._object(class_ids[i], scores[i], x1[i], y1[i], x2[i], y2[i])
                   for i in self._kept(class_ids, scores, conf, cls)]
        return objects[:max_det] if max_det else objects


class SEG(_RFDETR):
    """
    Instance segmentation: RF-DETR Seg — each object's box and outline.

    The ONNX's "masks" (1, Q, Hm, Wm) are mask logits at a fraction of the
    input; an object's mask is its logits upsampled to the image (bilinear,
    half-pixel centres, edges clamped — RF-DETR's upsampling) and
    thresholded at 0. Its outline is the mask's largest external contour,
    every corner kept (OpenCV's CHAIN_APPROX_SIMPLE drops only the points
    inside straight runs — lossless).

    Call:
        SEG(img, conf=0.3, cls=None, max_det=None) ->
        list[SimpleNamespace(prob, cls, rect, color, polygon)]

    `polygon` is [[x, y], ...] in original-image pixel coords; [] when the
    object's mask is empty.
    """

    def __call__(self, img, conf=0.3, cls=None, max_det=None, **kwargs):
        if img is None or img.size == 0:
            return []
        H, W = img.shape[:2]
        out = self.model(self._preprocess(img))
        queries, class_ids, scores = self._ranked(out["labels"][0])
        x1, y1, x2, y2 = self._boxes(out["dets"][0], queries, W, H)
        objects = []
        for i in self._kept(class_ids, scores, conf, cls)[:max_det or None]:
            obj = self._object(class_ids[i], scores[i], x1[i], y1[i], x2[i], y2[i])
            obj.polygon = self._outline(out["masks"][0][queries[i]], W, H)
            objects.append(obj)
        return objects

    @staticmethod
    def _outline(logits, W, H):
        """One object's outline: its mask logits upsampled to W x H, thresholded at 0."""
        mask = cv2.resize(logits, (W, H), interpolation=cv2.INTER_LINEAR) > 0
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float).tolist()


class KP(_RFDETR):
    """
    Keypoint detection: RF-DETR keypoints — every object and its keypoints
    in one pass.

    Pickle (beyond _RFDETR's): keypoint_names {class: [name, ...]} and
    skeleton {class: [(i, j), ...]} (1-indexed, COCO's convention) for the
    classes that have keypoints; meta num_keypoints_per_class (per logit
    slot, 0 = none) and trace_alpha. The ONNX's "keypoints" (1, Q, slots·K, D)
    hold, per query and class slot, K keypoints of x, y (0..1), a
    visibility logit, and — D ≥ 7 — the Cholesky factors of each
    keypoint's precision.

    RF-DETR's own decode: the ranked (query, class) pairs; each pair's
    keypoints from its class slot; and, when the model carries precisions
    and trace_alpha > 0, the object's score fused with its keypoints'
    uncertainty — sigmoid(log score − trace_alpha · log mean trace), the
    score `conf` is compared with.

    Call:
        KP(img, conf=0.3, kp_conf=0.5, cls=None, max_det=None) ->
        list[SimpleNamespace(prob, cls, rect, color, keypoints)]

    `keypoints` is [SimpleNamespace(name, x, y, conf, color), ...] in
    original-image pixel coords: the object's keypoints whose visibility
    clears `kp_conf`. skeleton_names(cls) gives its class's edges.
    """

    _PALETTE = [(51, 153, 255), (0, 255, 0), (255, 128, 0), (255, 51, 255),
                (0, 255, 255), (255, 255, 0), (128, 0, 255), (255, 0, 128)]

    def _load(self, data):
        meta = data["meta"]
        self.num_keypoints = [int(n) for n in meta["num_keypoints_per_class"]]
        self.trace_alpha = float(meta["trace_alpha"])
        self.keypoint_names = dict(data["keypoint_names"])
        self.skeleton = {c: [tuple(e) for e in edges] for c, edges in data.get("skeleton", {}).items()}
        for slot, name in enumerate(self.cls):
            if name is not None and slot < len(self.num_keypoints) and self.num_keypoints[slot] > 0:
                if len(self.keypoint_names.get(name, ())) != self.num_keypoints[slot]:
                    raise ValueError("Invalid model: class %r has %d keypoints, keypoint_names lists %d — "
                                     "re-export it from its training notebook."
                                     % (name, self.num_keypoints[slot], len(self.keypoint_names.get(name, ()))))

    def keypoint_color(self, cls, name):
        """A keypoint's colour: the pickle's colors[name], else a fixed palette by its index."""
        if isinstance(self.colors, dict) and name in self.colors:
            return _hex_to_bgr(self.colors[name])
        return self._PALETTE[self.keypoint_names[cls].index(name) % len(self._PALETTE)]

    def skeleton_names(self, cls):
        """The class's skeleton as keypoint-name pairs (the pickle's are 1-indexed)."""
        names = self.keypoint_names.get(cls, [])
        return [(names[i - 1], names[j - 1]) for i, j in self.skeleton.get(cls, [])]

    @staticmethod
    def _log_mean_trace(kps):
        """RF-DETR's keypoint uncertainty: the visibility-weighted mean of
        log trace(Σ) over an object's keypoints, Σ from each keypoint's
        precision Cholesky factors (log l11, l21, log l22)."""
        log_t1, log_t2 = -2.0 * kps[:, 4], -2.0 * kps[:, 6]
        log_t3 = 2.0 * np.log(np.maximum(np.abs(kps[:, 5]), 1e-12)) + log_t1 + log_t2
        log_trace = np.logaddexp.reduce(np.stack([log_t1, log_t2, log_t3], axis=-1), axis=-1)
        log_w = np.log(np.maximum(_sigmoid(kps[:, 2]), 1e-12))
        return np.logaddexp.reduce(log_trace + log_w) - np.logaddexp.reduce(log_w)

    def __call__(self, img, conf=0.3, kp_conf=0.5, cls=None, max_det=None, **kwargs):
        if img is None or img.size == 0:
            return []
        H, W = img.shape[:2]
        out = self.model(self._preprocess(img))
        queries, class_ids, scores = self._ranked(out["labels"][0])
        kps_all = out["keypoints"][0]
        slots, max_kp = len(self.num_keypoints), max(self.num_keypoints)
        kps_all = kps_all.reshape(kps_all.shape[0], slots, max_kp, kps_all.shape[-1])

        decoded = {}
        scores = scores.astype(np.float64)
        for i in range(len(queries)):
            c = int(class_ids[i])
            if c >= slots or self.num_keypoints[c] <= 0:
                continue
            kps = kps_all[queries[i], c, :self.num_keypoints[c]].astype(np.float64)
            decoded[i] = kps
            if self.trace_alpha > 0 and kps.shape[-1] >= 7:
                fused = _sigmoid(np.log(scores[i]) - self.trace_alpha * self._log_mean_trace(kps))
                scores[i] = min(fused, np.nextafter(1.0, 0.0))

        order = np.argsort(-scores, kind="stable")
        queries, class_ids, scores = queries[order], class_ids[order], scores[order]
        decoded = {new: decoded[old] for new, old in enumerate(order) if old in decoded}
        x1, y1, x2, y2 = self._boxes(out["dets"][0], queries, W, H)

        objects = []
        for i in self._kept(class_ids, scores, conf, cls)[:max_det or None]:
            obj = self._object(class_ids[i], scores[i], x1[i], y1[i], x2[i], y2[i])
            names = self.keypoint_names.get(obj.cls, [])
            obj.keypoints = []
            for k, kp in enumerate(decoded.get(i, ())):
                visibility = float(_sigmoid(kp[2]))
                if visibility >= kp_conf:
                    obj.keypoints.append(SimpleNamespace(
                        name=names[k], x=float(kp[0] * W), y=float(kp[1] * H), conf=visibility,
                        color=self.keypoint_color(obj.cls, names[k])))
            objects.append(obj)
        return objects


# ───────────────────────────────────────────────────────────────────────
# CLS — Classification (DINOv3 + a linear head)
# ───────────────────────────────────────────────────────────────────────

class CLS(object):
    """
    Image classifier (softmax over named classes): DINOv3 ViT-S/16 features
    and a linear head, one graph, on the GPU with TensorRT.

    Pickle:
        onnx   : bytes — input "input" (1, 3, H, W): the centred letterbox;
                 output "logits" (1, C)
        cls    : list[str]
        colors : dict|list of hex strings
        meta   : {type: "cls", image_size: (W, H), arch: "dinov3-linear", model,
                  precision: "bf16", input}

    Call:
        CLS(img, conf=0.5, top_k=1) -> list[[name, prob, color], ...]
        # legacy shape: a list of 3-element lists, sorted by prob desc.

    Empty list when no class clears `conf`. `top_k` caps the result.
    """

    def __init__(self, path, **kwargs):
        data = _load_pickle(path)
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
        # RGB and the mean/std happen in the engine (_image_input)
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
# OCR — PP-OCRv6 (text detection + recognition)
# ───────────────────────────────────────────────────────────────────────

# PaddleOCR's own OCR models — PP-OCRv6 medium, what its OCR pipeline runs
# (Apache-2.0) — pinned to a Hugging Face revision and the ONNX's sha256.
# fetch_ocr() downloads them into the cache as model pickles and prepares
# their engines, once per unit (the upgrade runs it). PaddleOCR's optional
# text-line orientation and document-unwarping models are not used.
_OCR_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dorna_vision", "ocr")
_OCR_MODELS = {
    "det": ("PaddlePaddle/PP-OCRv6_medium_det_onnx", "61323801669c338b7891481ec7bac61ce31b576a",
            "eb13b44b25bb36f89528b68720af8a61d9cf381176107f465db1757b65d086e1"),
    "rec": ("PaddlePaddle/PP-OCRv6_medium_rec_onnx", "50c7eacafc52fa7bcf4194e8cd08e46f8558504b",
            "9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba"),
}


def _ocr_model_dict(part, onnx_bytes, config):
    """The pickle of one PP-OCRv6 model. Detection: the image at its own
    size (a multiple of 32, up to 4000), BGR with ImageNet mean/std — as
    PaddleOCR normalises it. Recognition: one text line, 48 high, 320 to
    3200 wide, as a float (x/255 − 0.5)/0.5 the caller pads; its
    characters are the model's dictionary (CTC: blank first, space last)."""
    name = config["Global"]["model_name"]
    if part == "det":
        return {"onnx": onnx_bytes, "meta": {
            "type": "ocr_det", "model": name, "precision": "fp16",
            "input": {"format": "uint8-bgr", "channels": "bgr",
                      "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "profile": {"min": [1, 3, 32, 32], "opt": [1, 3, 736, 736], "max": [1, 3, 4000, 4000]}}}
    return {"onnx": onnx_bytes, "characters": list(config["PostProcess"]["character_dict"]), "meta": {
        "type": "ocr_rec", "model": name, "precision": "fp16", "input": {"format": "float"},
        "profile": {"min": [1, 3, 48, 320], "opt": [1, 3, 48, 320], "max": [1, 3, 48, 3200]}}}


def fetch_ocr():
    """
    Download PP-OCRv6 (pinned revision, checksum verified) into the cache
    as model pickles and prepare their engines — once per unit, minutes;
    after that, nothing. Returns {part: prepare()'s result}.
    """
    import urllib.request
    import yaml
    os.makedirs(_OCR_DIR, exist_ok=True)
    prepared = {}
    for part, (repo, revision, sha256) in _OCR_MODELS.items():
        path = os.path.join(_OCR_DIR, part + ".pkl")
        if not os.path.exists(path):
            base = "https://huggingface.co/%s/resolve/%s/" % (repo, revision)
            onnx_bytes = urllib.request.urlopen(base + "inference.onnx").read()
            if hashlib.sha256(onnx_bytes).hexdigest() != sha256:
                raise ValueError("%s inference.onnx: checksum mismatch — refusing it" % repo)
            config = yaml.safe_load(urllib.request.urlopen(base + "inference.yml").read())
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(_ocr_model_dict(part, onnx_bytes, config), f)
            os.replace(tmp, path)
        prepared[part] = prepare(path)
    return prepared


class OCR(object):
    """
    Text detection + recognition: PaddleOCR's PP-OCRv6 pipeline on TensorRT,
    its steps and defaults as PaddleOCR runs them (its pipeline config
    OCR.yaml, its processors):

      detect   the image resized to multiples of 32 (sides of 64 or more,
               up to 4000); the DB probability map > 0.3; each contour's
               minimum-area rectangle, kept when its mean probability
               reaches 0.6, grown by unclip ratio 1.5, at least 5 px across
      order    top to bottom, left to right within 10 px
      crop     each box's rectangle, perspective-warped straight (cubic),
               turned upright when it is 1.5x taller than wide
      read     each line on its own: 48 high, at least 320 wide (zeros
               after it), CTC decoded — its score the mean of its
               characters' probabilities. (PaddleOCR batches six lines
               padded to the widest; one at a time, a line's text never
               depends on the others in the image.)

    Call:
        OCR().ocr(img, conf=0.5, detection_enabled=True) ->
        list[[corners, [text, score]], ...]   # reading order; score ≥ conf

    `corners` are the line's 4 points (top-left, top-right, bottom-right,
    bottom-left) in image pixels. detection_enabled=False reads the whole
    image as one line.
    """

    DET_LIMIT_SIDE = 64          # PaddleOCR's OCR.yaml: limit_side_len, limit_type "min"
    DET_MAX_SIDE = 4000          # max_side_limit
    DET_THRESH = 0.3             # thresh
    DET_BOX_THRESH = 0.6         # box_thresh
    DET_UNCLIP = 1.5             # unclip_ratio
    DET_MAX_CANDIDATES = 3000    # the model's own (inference.yml)
    DET_MIN_SIZE = 3             # DBPostProcess
    REC_HEIGHT, REC_MIN_WIDTH, REC_MAX_WIDTH = 48, 320, 3200

    def __init__(self, **kwargs):
        paths = {part: os.path.join(_OCR_DIR, part + ".pkl") for part in _OCR_MODELS}
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            raise ModelNotPreparedError("OCR models are not on this unit: run dorna_vision.ai.fetch_ocr() "
                                        "once (the upgrade does) — missing %s" % ", ".join(missing))
        rec = _load_pickle(paths["rec"])
        self.characters = ["blank"] + rec["characters"] + [" "]
        self.det = _TRTModel(_load_pickle(paths["det"]))
        self.rec = _TRTModel(rec)

    def __del__(self):
        for name in ("det", "rec"):
            model = getattr(self, name, None)
            if model is not None:
                model.close()
                setattr(self, name, None)

    # ── detect ──
    def _det_input(self, img):
        """PaddleOCR's DetResizeForTest (limit type "min"): (resized image,
        its height and width ratios to the original)."""
        h, w = img.shape[:2]
        if h + w < 64:                                   # its image_padding
            padded = np.zeros((max(32, h), max(32, w), 3), np.uint8)
            padded[:h, :w] = img
            img = padded
            h, w = img.shape[:2]
        ratio = float(self.DET_LIMIT_SIDE) / min(h, w) if min(h, w) < self.DET_LIMIT_SIDE else 1.0
        rh, rw = int(h * ratio), int(w * ratio)
        if max(rh, rw) > self.DET_MAX_SIDE:
            ratio = float(self.DET_MAX_SIDE) / max(rh, rw)
            rh, rw = int(rh * ratio), int(rw * ratio)
        rh, rw = max(int(round(rh / 32) * 32), 32), max(int(round(rw / 32) * 32), 32)
        if (rh, rw) != (h, w):
            img = cv2.resize(img, (rw, rh))
        return img

    @staticmethod
    def _mini_box(contour):
        """The contour's minimum-area rectangle, its corners ordered
        top-left, top-right, bottom-right, bottom-left; and its short side."""
        rect = cv2.minAreaRect(contour)
        p = sorted(list(cv2.boxPoints(rect)), key=lambda x: x[0])
        a, d = (0, 1) if p[1][1] > p[0][1] else (1, 0)
        b, c = (2, 3) if p[3][1] > p[2][1] else (3, 2)
        return np.array([p[a], p[b], p[c], p[d]]), min(rect[1])

    @staticmethod
    def _box_score(prob, box):
        """The mean probability inside the box (DBPostProcess's fast score)."""
        h, w = prob.shape
        xmin = max(0, min(math.floor(box[:, 0].min()), w - 1))
        xmax = max(0, min(math.ceil(box[:, 0].max()), w - 1))
        ymin = max(0, min(math.floor(box[:, 1].min()), h - 1))
        ymax = max(0, min(math.ceil(box[:, 1].max()), h - 1))
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        local = box.copy()
        local[:, 0] -= xmin
        local[:, 1] -= ymin
        cv2.fillPoly(mask, local.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(prob[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def _unclip(self, box):
        """The box grown by area·ratio/perimeter (pyclipper, round joins)."""
        import pyclipper
        distance = cv2.contourArea(box) * self.DET_UNCLIP / cv2.arcLength(box, True)
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        return np.array(offset.Execute(distance)[0])

    def _detect(self, img):
        H, W = img.shape[:2]
        prob = self.det(self._det_input(img)[None])
        prob = next(iter(prob.values()))[0, 0]
        ph, pw = prob.shape
        contours, _ = cv2.findContours(((prob > self.DET_THRESH) * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours[:self.DET_MAX_CANDIDATES]:
            box, side = self._mini_box(contour)
            if side < self.DET_MIN_SIZE or self._box_score(prob, box.reshape(-1, 2)) < self.DET_BOX_THRESH:
                continue
            box, side = self._mini_box(self._unclip(box).reshape(-1, 1, 2))
            if side < self.DET_MIN_SIZE + 2:
                continue
            box[:, 0] = np.clip(np.round(box[:, 0] * W / pw), 0, W)
            box[:, 1] = np.clip(np.round(box[:, 1] * H / ph), 0, H)
            boxes.append(box.astype(np.int16))
        # reading order (SortQuadBoxes): top to bottom; within 10 px, left to right
        boxes = sorted(boxes, key=lambda b: (b[0][1], b[0][0]))
        for i in range(len(boxes) - 1):
            for j in range(i, -1, -1):
                if abs(boxes[j + 1][0][1] - boxes[j][0][1]) < 10 and boxes[j + 1][0][0] < boxes[j][0][0]:
                    boxes[j], boxes[j + 1] = boxes[j + 1], boxes[j]
                else:
                    break
        return boxes

    # ── read ──
    @staticmethod
    def _crop(img, box):
        """The box's minimum-area rectangle, warped straight; upright when tall."""
        rect, _ = OCR._mini_box(np.array(box).astype(np.int32))
        rect = rect.astype(np.float32)
        width = int(max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3])))
        height = int(max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2])))
        target = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        crop = cv2.warpPerspective(img, cv2.getPerspectiveTransform(rect, target), (width, height),
                                   borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
        if crop.shape[0] * 1.0 / max(crop.shape[1], 1) >= 1.5:
            crop = np.rot90(crop)
        return crop

    def _read(self, crop):
        """One line: (text, score)."""
        h, w = crop.shape[:2]
        width = int(self.REC_HEIGHT * max(self.REC_MIN_WIDTH / self.REC_HEIGHT, w / float(h)))
        if width > self.REC_MAX_WIDTH:
            resized, used = cv2.resize(crop, (self.REC_MAX_WIDTH, self.REC_HEIGHT)), self.REC_MAX_WIDTH
            width = self.REC_MAX_WIDTH
        else:
            used = min(int(math.ceil(self.REC_HEIGHT * w / float(h))), width)
            resized = cv2.resize(crop, (used, self.REC_HEIGHT))
        line = np.zeros((1, 3, self.REC_HEIGHT, width), dtype=np.float32)
        line[0, :, :, :used] = (resized.astype(np.float32).transpose(2, 0, 1) / 255 - 0.5) / 0.5
        probs = next(iter(self.rec(line).values()))[0]
        index, prob = probs.argmax(-1), probs.max(-1)
        keep = np.ones(len(index), dtype=bool)
        keep[1:] = index[1:] != index[:-1]                 # merge repeats
        keep &= index != 0                                 # drop CTC blanks
        text = "".join(self.characters[i] for i in index[keep])
        return text, float(np.mean(prob[keep])) if keep.any() else 0.0

    def ocr(self, img, conf=0.5, detection_enabled=True, **kwargs):
        if img is None or img.size == 0:
            return []
        h, w = img.shape[:2]
        if not detection_enabled:
            text, score = self._read(img)
            return [[[[0, 0], [w, 0], [w, h], [0, h]], [text, score]]] if score >= conf else []
        results = []
        for box in self._detect(img):
            crop = self._crop(img, box)
            if crop.size == 0:
                continue
            text, score = self._read(crop)
            if score >= conf:
                results.append([box.tolist(), [text, score]])
        return results
