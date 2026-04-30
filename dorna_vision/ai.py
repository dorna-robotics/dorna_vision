"""
ML inference primitives for the dorna_vision pipeline.

Each class is a single-purpose, init-once / call-many wrapper around an
OpenVINO IR model packaged in a pickle. The pickle schema (produced by the
training notebooks under draft/) is:

    {
        "xml":     "<openvino IR XML, str>",
        "bin":     <openvino IR weights, bytes>,
        "cls":     ["class_a", "class_b", ...]      # for OD/ROD/CLS/ANOM
        "colors":  {"class_a": "#aabbcc", ...}      # dict OR list of hex
        "meta":    {"image_size": ..., ...},        # see each class

        # KP-only:
        "keypoint_names": [...],
        "skeleton":       [(1,2), ...]
    }

The new classes follow the inference paths in:
    draft/mmdetection/inference_custom    -> OD
    draft/mmrotate/inference_custom       -> ROD
    draft/mmpose/inference_custom         -> KP
    draft/cls/inference_classification    -> CLS
    draft/anomaly/stfpm/inference_anomaly -> ANOM

Composition: OD/ROD give you boxes; crop the image around a box (with
padding) and feed the crop to KP. CLS labels a whole image (or a crop).
ANOM scores a whole image and returns hot-spots.

Legacy implementations are kept as OD_v1, CLS_v1, KP_v1 (and the older
NCNN ones as OD_NCNN, CLS_NCNN) for back-compat — these will be retired
once existing pickles are migrated.
"""

import math
import os
import pickle
import importlib
from types import SimpleNamespace

import cv2
import numpy as np
from openvino import Core

import ncnn
from ncnn.utils.objects import Detect_Object

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
    dict (used internally by the legacy KP_v1 wrapper).
    """
    if isinstance(path, dict):
        return path
    with open(path, "rb") as f:
        return pickle.load(f)


def _read_openvino_from_pickle(model_dict, input_shape, device_name="CPU", device_config=None):
    """
    Build a compiled OpenVINO model from {xml, bin} pickle entries.
    `input_shape` is a list passed to model.reshape(). Returns
    (core, model, compiled_model).
    """
    xml_data = model_dict["xml"]
    bin_data = model_dict["bin"]
    core = Core()
    model = core.read_model(
        model=bytes(xml_data, "utf-8") if isinstance(xml_data, str) else xml_data,
        weights=bin_data,
    )
    if input_shape is not None:
        model.reshape({model.input(0): input_shape})
    cfg = device_config if device_config is not None else _DEVICE_CONFIG
    compiled = core.compile_model(model=model, device_name=device_name, config=cfg)
    return core, model, compiled


def _release_openvino(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass


def _bbox_to_corners(x, y, w, h):
    return [
        [x,         y],
        [x + w,     y],
        [x + w,     y + h],
        [x,         y + h],
    ]


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


# ═══════════════════════════════════════════════════════════════════════
# Legacy implementations — preserved for back-compat with existing pickles.
# Slated for removal once consumers have migrated to the new classes above.
# ═══════════════════════════════════════════════════════════════════════

def letterbox_image(image, target_size=(640, 640), pad_color=(128, 128, 128)):
    """Legacy letterbox helper used by OD_v1 / CLS_v1 / OCR."""
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


def hex_to_bgr(hex_color):
    """Legacy alias kept for code that imported it directly."""
    return _hex_to_bgr(hex_color)


class KP_v1(object):
    """Legacy bundled OD+KP (one pickle, both models). Pre-mmdetection format."""
    def __init__(self, path=None, device_name="CPU", **kwargs):
        with open(path, "rb") as file:
            data = pickle.load(file)
        self.detection = {
            "od": OD_v1(path=None, device_name=device_name,
                        data={"bin": data["bin"], "xml": data["xml"], "cls": data["cls"],
                              "colors": data["colors"], "meta": data["meta"]}),
            "kp": {k: OD_v1(path=None, device_name=device_name, data=data["kp"][k])
                   for k in data["kp"] if data["kp"][k]},
        }

    def od(self, img, conf=0.5, cls=[], **kwargs):
        return self.detection["od"](img, conf=conf, cls=cls, **kwargs)

    def kp(self, img, label, bb, offset=20, conf=0.5, cls=[], **kwargs):
        retval = []
        roi = ROI(img, corners=bb, crop=True, offset=offset)
        if label not in self.detection["kp"]:
            return []
        valid_cls = list(self.detection["kp"][label].cls)
        if not cls:
            valid_cls = [c for c in cls if c in valid_cls]
        retval = self.detection["kp"][label](roi.img, conf=conf, cls=valid_cls)
        for r in retval:
            r.center = roi.pxl_to_orig([r.rect.x + r.rect.w / 2, r.rect.y + r.rect.h / 2])
        return retval

    def __del__(self):
        try:
            self.detection["od"].__del__()
            for k in self.detection["kp"]:
                self.detection["kp"][k].__del__()
        except Exception:
            pass


class OD_v1(object):
    """Legacy YOLOX-style OpenVINO object detector."""
    def __init__(self, path=None, device_name="CPU", **kwargs):
        if path is not None:
            with open(path, "rb") as file:
                data = pickle.load(file)
        else:
            data = kwargs["data"]

        self.cls = data["cls"]
        self.colors = {k: hex_to_bgr(data["colors"][k]) for k in data["colors"]}

        self.core = Core()
        self.model = self.core.read_model(model=bytes(data["xml"], "utf-8"), weights=bytes(data["bin"]))
        self.compiled_model = self.core.compile_model(model=self.model, device_name=device_name)
        self.input_shape = self.compiled_model.input(0).shape

    def __del__(self):
        try:
            del self.model
            del self.compiled_model
            self.core = None
        except Exception:
            pass

    def _postprocess(self, outputs, img_size, p6=False):
        grids = []
        expanded_strides = []
        strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]
        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]
        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))
        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        return outputs

    def _nms(self, boxes, scores, nms_thr):
        x1 = boxes[:, 0]; y1 = boxes[:, 1]
        x2 = boxes[:, 2]; y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]
        return keep

    def _multiclass_nms(self, boxes, scores, nms_thr, score_thr):
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self._nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1)
        return dets

    def __call__(self, img, conf=0.5, cls=[], max_det=None, nms_thr=0.3, **kwargs):
        objects = []
        fixed_size = (self.input_shape[2], self.input_shape[3])

        resized_img, scale, pad_w, pad_h = letterbox_image(
            img, target_size=(fixed_size[1], fixed_size[0]), pad_color=(128, 128, 128)
        )
        preprocessed_img = resized_img.transpose((2, 0, 1))
        preprocessed_img = np.expand_dims(preprocessed_img, axis=0).astype(np.float32)

        output = self.compiled_model([preprocessed_img])[self.compiled_model.output(0)]
        predictions = self._postprocess(output, fixed_size)[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4, None] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.

        boxes_xyxy[:, [0, 2]] -= pad_w
        boxes_xyxy[:, [1, 3]] -= pad_h
        boxes_xyxy /= scale

        dets = self._multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=conf)
        if dets is None:
            return objects

        final_boxes = dets[:, :4].tolist()
        final_scores = dets[:, 4].tolist()
        final_cls_inds = dets[:, 5].tolist()
        for i in range(len(final_boxes)):
            score = final_scores[i]
            if score < conf:
                continue
            label = self.cls[int(final_cls_inds[i])]
            if cls and label not in cls:
                continue
            obj = SimpleNamespace(
                prob=float(score),
                cls=label,
                rect=SimpleNamespace(
                    x=final_boxes[i][0],
                    y=final_boxes[i][1],
                    w=final_boxes[i][2] - final_boxes[i][0],
                    h=final_boxes[i][3] - final_boxes[i][1],
                ),
                color=self.colors[label],
            )
            objects.append(obj)

        objects = sorted(objects, key=lambda x: x.prob, reverse=True)
        if max_det:
            objects = objects[:max_det]
        return objects


class CLS_v1(object):
    """Legacy OpenVINO classifier (ImageNet-mean preprocessing)."""
    def __init__(self, path, target_size=(224, 224), device_name="CPU", **kwargs):
        with open(path, "rb") as file:
            data = pickle.load(file)
        self.cls = data["cls"]
        self.colors = {k: hex_to_bgr(data["colors"][k]) for k in data["colors"]}

        self.core = Core()
        self.model = self.core.read_model(model=bytes(data["xml"], "utf-8"), weights=bytes(data["bin"]))
        self.compiled_model = self.core.compile_model(model=self.model, device_name=device_name)

        self.target_size = target_size
        self.input_tensor = self.compiled_model.input(0)
        self.output_tensor = self.compiled_model.output(0)
        self.mean = np.array([123.675, 116.28, 103.53])
        self.scale = np.array([58.395, 57.12, 57.375])

    def __del__(self):
        try:
            del self.model
            del self.compiled_model
            self.core = None
        except Exception:
            pass

    def __call__(self, img, conf=0.5, **kwargs):
        retval = []
        img_resized, _, _, _ = letterbox_image(img, self.target_size)
        img_normalized = img_resized.astype(np.float32)
        for i in range(3):
            img_normalized[:, :, i] = (img_normalized[:, :, i] - self.mean[i]) / self.scale[i]
        img_chw = img_normalized.transpose(2, 0, 1)
        input_data = np.expand_dims(img_chw, axis=0)
        results = self.compiled_model([input_data])
        output_data = results[self.output_tensor]
        exp_scores = np.exp(output_data)
        softmax_probs = exp_scores / np.sum(exp_scores)
        predicted_class = np.argmax(softmax_probs)
        probability = softmax_probs[0][predicted_class]
        if probability > conf:
            retval = [
                [self.cls[predicted_class], float(probability), self.colors[self.cls[predicted_class]]]
            ]
        return retval


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


class CLS_NCNN(object):
    def __init__(self, path, target_size=224, num_threads=2, use_gpu=False, **kwargs):
        self.path = path
        self.target_size = target_size
        self.num_threads = num_threads
        self.use_gpu = use_gpu
        self.mean_vals = []
        self.norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu
        self.net.opt.num_threads = self.num_threads

        with open(self.path, "rb") as file:
            data = pickle.load(file)
        with open("temp.param", "w", encoding="utf-8") as f:
            f.write(data["param"])
        with open("temp.bin", "wb") as f:
            f.write(data["bin"])
        self.net.load_param("temp.param")
        self.net.load_model("temp.bin")
        self.cls = data["cls"]
        self.model = data["meta"]["model"]
        os.remove("temp.param")
        os.remove("temp.bin")

    def __del__(self):
        self.net = None

    def __call__(self, img, conf=0.5, **kwargs):
        retval = []
        if self.model.startswith("shufflenet_v2"):
            img_h = img.shape[0]; img_w = img.shape[1]
            mat_in = ncnn.Mat.from_pixels_resize(
                img, ncnn.Mat.PixelType.PIXEL_BGR,
                img_w, img_h, self.target_size, self.target_size,
            )
            mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)
            ex = self.net.create_extractor()
            ex.input("in0", mat_in)
            _, mat_out = ex.extract("out0")
            softmax = ncnn.create_layer("Softmax")
            pd = ncnn.ParamDict()
            softmax.load_param(pd)
            softmax.forward_inplace(mat_out, self.net.opt)
            mat_out = mat_out.reshape(mat_out.w * mat_out.h * mat_out.c)
            cls_scores = np.array(mat_out).astype(float).tolist()
            max_cls, max_score = max(zip(self.cls, cls_scores), key=lambda x: x[1])
            if max_score >= conf:
                retval = [[max_cls, max_score]]
        return retval


class OD_NCNN(object):
    def __init__(self, path, target_size=416, num_threads=2, use_gpu=False, **kwargs):
        self.path = path
        self.target_size = target_size
        self.num_threads = num_threads
        self.use_gpu = use_gpu
        self.mean_vals = []
        self.norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu
        self.net.opt.num_threads = self.num_threads

        with open(self.path, "rb") as file:
            data = pickle.load(file)
        with open("temp.param", "w", encoding="utf-8") as f:
            f.write(data["param"])
        with open("temp.bin", "wb") as f:
            f.write(data["bin"])
        self.net.load_param("temp.param")
        self.net.load_model("temp.bin")
        self.cls = data["cls"]
        self.model = data["meta"]["model"]
        os.remove("temp.param")
        os.remove("temp.bin")

    def __del__(self):
        self.net = None

    def __call__(self, img, conf=0.5, cls=[], max_det=None, **kwargs):
        objects = []
        if self.model.startswith("yolov4"):
            img_h = img.shape[0]; img_w = img.shape[1]
            mat_in = ncnn.Mat.from_pixels_resize(
                img, ncnn.Mat.PixelType.PIXEL_BGR2RGB,
                img_w, img_h, self.target_size, self.target_size,
            )
            mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)
            ex = self.net.create_extractor()
            ex.input("data", mat_in)
            _, mat_out = ex.extract("output")
            for i in range(mat_out.h):
                values = mat_out.row(i)
                obj = Detect_Object()
                obj.prob = values[1]
                if obj.prob < conf:
                    continue
                obj.cls = self.cls[int(values[0] - 1)]
                if cls and obj.cls not in cls:
                    continue
                obj.rect.x = values[2] * img_w
                obj.rect.y = values[3] * img_h
                obj.rect.w = values[4] * img_w - obj.rect.x
                obj.rect.h = values[5] * img_h - obj.rect.y
                objects.append(obj)
            objects = sorted(objects, key=lambda x: x.prob, reverse=True)
            if max_det:
                objects = objects[:max_det]
        return objects
