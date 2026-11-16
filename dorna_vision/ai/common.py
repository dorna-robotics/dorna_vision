"""
What both engines share: colours, the centred letterbox, loading a model
pickle, the keypoint naming and colouring, and the error a model that is
not ready on this unit raises.
"""

import pickle

import cv2
import numpy as np


_PAD_VALUE = 128


class ModelNotPreparedError(ValueError):
    """The model is not ready on this unit: prepare() it first."""


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
    """Load a model pickle. Accepts a filesystem path or a pre-loaded dict."""
    if isinstance(path, dict):
        return path
    with open(path, "rb") as f:
        return pickle.load(f)


class _Keypoints(object):
    """
    The keypoint half of a KP model, the same on both engines: each object
    class's keypoint names ({class: [name, ...]}) and skeleton
    ({class: [(i, j), ...]}, 1-indexed — COCO's convention), and a
    keypoint's colour.
    """

    _PALETTE = [(51, 153, 255), (0, 255, 0), (255, 128, 0), (255, 51, 255),
                (0, 255, 255), (255, 255, 0), (128, 0, 255), (255, 0, 128)]

    def keypoint_color(self, cls, name):
        """A keypoint's colour: the pickle's colors[name], else a fixed palette by its index."""
        if isinstance(self.colors, dict) and name in self.colors:
            return _hex_to_bgr(self.colors[name])
        return self._PALETTE[self.keypoint_names[cls].index(name) % len(self._PALETTE)]

    def skeleton_names(self, cls):
        """The class's skeleton as keypoint-name pairs (the pickle's are 1-indexed)."""
        names = self.keypoint_names.get(cls, [])
        return [(names[i - 1], names[j - 1]) for i, j in self.skeleton.get(cls, [])]
