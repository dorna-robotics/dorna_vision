"""
The OpenVINO engine — the Pi's CPU. Each class is a single-purpose,
init-once / call-many wrapper around an OpenVINO IR model packaged in a
pickle (written by the training notebooks under
training_notebooks/pi/):

    {
        "xml":     "<openvino IR XML, str>",
        "bin":     <openvino IR weights, bytes>,
        "cls":     ["class_a", "class_b", ...]
        "colors":  {"class_a": "#aabbcc", ...}      # dict OR list of hex
        "meta":    {"type": ..., "image_size": ..., ...},   # see each class

        # KP-only:
        "keypoint_names": [...],
        "skeleton":       [(1,2), ...]
    }

    OD    — mmdetection: boxes
    ROD   — mmrotate: rotated boxes
    KP    — RTMPose, top-down: one object's keypoints, in a box given to it
    CLS   — one label per image
    ANOM  — STFPM: a pass/fail verdict per image
    OCR   — horizontal-text detection + PP-OCR recognition (built in)

Load them through dorna_vision.ai, which picks the engine from the model.
OpenVINO compiles a model as it loads: there is nothing to prepare.
"""

import math
import os
import importlib.util
import threading
from types import SimpleNamespace

import cv2
import numpy as np
from openvino import Core

from dorna_vision.ai.common import (_PAD_VALUE, _Keypoints, _color_for_class,
                                    _letterbox_centered, _load_pickle)


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
# OD — Object detection (mmdetection / draft pickle format)
# ───────────────────────────────────────────────────────────────────────

class OD(object):
    """
    Axis-aligned object detector. Single forward pass, per-class NMS.

    Pickle:
        xml, bin                 — OpenVINO IR
        cls    : list[str]
        colors : dict|list of hex strings
        meta   : {image_size: int, model_type: str, precision: str}

    Call:
        OD(img, conf=0.3, nms_iou=0.5, cls=None, max_det=None) ->
        list[SimpleNamespace(prob, cls, rect, color)]

    `rect` is a SimpleNamespace(x, y, w, h) in original-image pixel coords.
    """

    def __init__(self, path, device_name="CPU", device_config=None, **kwargs):
        data = _load_pickle(path)
        self.cls    = list(data["cls"])
        self.colors = data.get("colors", {})
        meta = data.get("meta", {})
        self.image_size = int(meta.get("image_size", 640))

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
    def _nms_per_class(boxes_xyxy, scores, class_ids, iou_thr):
        keep = []
        for cid in np.unique(class_ids):
            idx = np.where(class_ids == cid)[0]
            if len(idx) == 0:
                continue
            b = boxes_xyxy[idx]
            s = scores[idx]
            xywh = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    for x1, y1, x2, y2 in b]
            picked = cv2.dnn.NMSBoxes(xywh, s.tolist(),
                                      score_threshold=0.0,
                                      nms_threshold=float(iou_thr))
            if len(picked) == 0:
                continue
            picked = np.array(picked).reshape(-1)
            keep.extend(idx[picked].tolist())
        return np.array(keep, dtype=np.int64)

    def __call__(self, img, conf=0.3, nms_thr=0.3, cls=None, max_det=None, **kwargs):
        # `nms_iou` accepted as a kwarg alias for back-compat with newer call sites.
        nms_thr = float(kwargs.pop("nms_iou", nms_thr))
        if img is None or img.size == 0:
            return []

        tensor, ratio = self._preprocess(img)
        results = self.compiled_model([tensor])
        raw = np.asarray(list(results.values())[0])   # (1, N, 4+1+C)
        pred = raw[0]

        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        obj = pred[:, 4]
        cls_scores = pred[:, 5:]

        class_ids = cls_scores.argmax(axis=1)
        max_cls   = cls_scores.max(axis=1)
        scores    = obj * max_cls

        keep_mask = scores > conf
        if not keep_mask.any():
            return []
        cx, cy, w, h = cx[keep_mask], cy[keep_mask], w[keep_mask], h[keep_mask]
        scores, class_ids = scores[keep_mask], class_ids[keep_mask]

        x1 = cx - w * 0.5
        y1 = cy - h * 0.5
        x2 = cx + w * 0.5
        y2 = cy + h * 0.5
        boxes = np.stack([x1, y1, x2, y2], axis=1) / ratio

        H, W = img.shape[:2]
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, W - 1)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, H - 1)

        keep_idx = self._nms_per_class(boxes, scores, class_ids, nms_thr)
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
            x1, y1, x2, y2 = boxes[i].tolist()
            objects.append(SimpleNamespace(
                prob=float(scores[i]),
                cls=label,
                rect=SimpleNamespace(x=float(x1), y=float(y1),
                                     w=float(x2 - x1), h=float(y2 - y1)),
                color=_color_for_class(class_ids[i], self.cls, self.colors),
            ))

        objects.sort(key=lambda o: o.prob, reverse=True)
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

class KP(_Keypoints):
    """
    Top-down keypoint detection (RTMPose): ONE object's keypoints, found in
    a box given to it — the whole image, or `bbox`. The box is framed as
    training framed it: its centre, the model's aspect ratio, `padding`
    (1.25 — 12.5 % more each side; 1.0 when the box is the whole image,
    which is the crop already).

    The result has the same shape as every KP model's: a list of objects,
    each with its keypoints. Here the object is the box's — the pickle's
    one class, the box as its rect — and its score is mmpose's for a given
    box: the mean confidence of the keypoints it reports.

    Pickle: xml, bin; cls [the object's class]; colors; keypoint_names
    [name, ...]; skeleton [(i, j), ...] (1-indexed); meta {type: "kp",
    input_w, input_h, ...}.

    Call:
        KP(img, conf=0.3, kp_conf=0.3, cls=None, bbox=None, padding=None) ->
        list[SimpleNamespace(prob, cls, rect, color, keypoints)]   # 0 or 1

    `keypoints` is [SimpleNamespace(name, x, y, conf, color), ...] in
    original-image pixel coords: those whose confidence clears `kp_conf`.
    The object is reported when its score clears `conf` (and it passes the
    three-state `cls` filter). `bbox` is (x, y, w, h) in image pixels.
    """

    top_down = True   # takes the box it looks in (detect.py passes bbox)

    def __init__(self, path, device_name="CPU", device_config=None, padding=1.25, **kwargs):
        data = _load_pickle(path)
        self.cls    = list(data["cls"])
        self.colors = data.get("colors", {})
        self.keypoint_names = {self.cls[0]: list(data["keypoint_names"])}
        self.skeleton       = {self.cls[0]: [tuple(e) for e in data.get("skeleton", [])]}
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
          2. bbox is None → 1.0 (the whole image is the crop already, so
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

    def __call__(self, img, conf=0.3, kp_conf=0.3, cls=None, bbox=None, padding=None, **kwargs):
        if img is None or img.size == 0:
            return []
        name = self.cls[0]
        if cls is not None and name not in set(cls):
            return []

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

        names = self.keypoint_names[name]
        keypoints = []
        for i in range(xy_orig.shape[0]):
            c = float(confs[i])
            if c < kp_conf:
                continue
            keypoints.append(SimpleNamespace(
                name=names[i],
                x=float(xy_orig[i, 0]),
                y=float(xy_orig[i, 1]),
                conf=c,
                color=self.keypoint_color(name, names[i]),
            ))

        score = float(np.mean([k.conf for k in keypoints])) if keypoints else 0.0
        if not keypoints or score < conf:
            return []
        H, W = img.shape[:2]
        x, y, w, h = (0, 0, W, H) if bbox is None else bbox
        return [SimpleNamespace(
            prob=score,
            cls=name,
            rect=SimpleNamespace(x=float(x), y=float(y), w=float(w), h=float(h)),
            color=_color_for_class(0, self.cls, self.colors),
            keypoints=keypoints,
        )]


# ───────────────────────────────────────────────────────────────────────
# CLS — Classification (cls / draft pickle format)
# ───────────────────────────────────────────────────────────────────────

class CLS(object):
    """
    Image classifier (softmax over named classes).

    Pickle:
        xml, bin
        cls    : list[str]
        colors : dict|list of hex strings
        meta   : {image_size: (W, H), type: "cls", model, precision}

    Call:
        CLS(img, conf=0.5, top_k=1) -> list[[name, prob, color], ...]
        # legacy shape: a list of 3-element lists, sorted by prob desc.

    Empty list when no class clears `conf`. `top_k` caps the result.
    """

    def __init__(self, path, device_name="CPU", device_config=None, **kwargs):
        data = _load_pickle(path)
        self.cls    = list(data["cls"])
        self.colors = data.get("colors", {})
        meta = data.get("meta", {})
        self.image_size = tuple(meta.get("image_size", (224, 224)))   # (W, H)

        self.core, self.model, self.compiled_model = _read_openvino_from_pickle(
            data,
            input_shape=[1, 3, self.image_size[1], self.image_size[0]],
            device_name=device_name,
            device_config=device_config,
        )

    def __del__(self):
        _release_openvino(getattr(self, "model", None), getattr(self, "compiled_model", None))
        self.core = None

    def _preprocess(self, img):
        padded = _letterbox_centered(img, self.image_size)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32)
        return ((rgb - _IMNET_MEAN) / _IMNET_STD).transpose(2, 0, 1)[None]

    @staticmethod
    def _softmax(x):
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def __call__(self, img, conf=0.5, top_k=1, **kwargs):
        if img is None or img.size == 0:
            return []
        tensor = self._preprocess(img)
        results = self.compiled_model([tensor])
        logits = np.asarray(list(results.values())[0]).reshape(-1)
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


# ───────────────────────────────────────────────────────────────────────
# OCR — horizontal-text detection + PP-OCR recognition (built-in models)
# ───────────────────────────────────────────────────────────────────────

def _letterbox_padded(image, target_size=(640, 640), pad_color=(128, 128, 128)):
    """Resize proportionally and centre-pad: (image, scale, pad_w, pad_h)."""
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
    """OpenVINO horizontal-text detection + recognition pipeline."""

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
            det_img, scale, pad_w, pad_h = _letterbox_padded(img, (640, 640))
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
