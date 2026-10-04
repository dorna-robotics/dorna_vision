"""A VLM detection's settings — the ``detection`` dict of a ``cmd: vlm``
detection, written inline or in a config file (vision-guide §9)::

    model: {name: flash-3.8, backend: default, thinking: low}
    output: cls            # cls | od | ocr — the same names as the other detections
    labels: [...]          # the allowed answers; [] = the model answers in its own words
    schema: {}             # JSON schema of extra ``data`` on every entry; {} = none
    prompt, references, image_size, timeout_s

By the time it reaches here the dict is final: the config file (if any)
is merged with the inline keys, the key is gone (it travels separately,
see ``preset_key``), and every reference image is ``{"b64": ...}``
(shipped by the client) or ``{"server": path}`` (a file on this unit).
Every field is required (the explicit-values rule); unknown fields are
refused. Reference images are decoded and resized ONCE; settings are
cached by a hash of their contents.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2 as cv
import numpy as np

# the vlm fields of a detection dict, besides "cmd"
FIELDS = ("model", "output", "labels", "schema", "prompt", "references",
          "image_size", "timeout_s")
MODEL_FIELDS = ("name", "backend", "thinking")   # model: {name, backend, thinking}
KEY_FIELDS = ("key", "key_path")        # resolved and removed before the server sees them
OUTPUTS = ("cls", "od", "ocr")          # the same names as the other detections
THINKING = ("minimal", "low", "medium", "high")


@dataclass
class Reference:
    label: str
    note: str
    jpeg: bytes                     # resized to image_size, JPEG


@dataclass
class Preset:
    name: str                       # the detection's name (recorded in its log)
    hash: str
    model: str                      # model.name — the backend's model name
    backend: str                    # model.backend
    thinking: str                   # model.thinking
    output: str                     # cls | od | ocr
    labels: List[str]               # the allowed answers; [] = open (the model's own words)
    schema: Dict[str, Any]          # JSON schema of extra ``data`` on every entry; {} = none
    prompt: str
    references: List[Reference]
    image_size: int
    timeout_s: float


def encode_image(img: np.ndarray, image_size: int) -> bytes:
    """Resize so the long side is at most ``image_size`` and encode JPEG —
    what every image (reference or view) is sent as."""
    if img is None or img.size == 0:
        raise ValueError("empty image")
    h, w = img.shape[:2]
    if max(h, w) > image_size:
        s = image_size / float(max(h, w))
        img = cv.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                        interpolation=cv.INTER_AREA)
    ok, buf = cv.imencode(".jpg", img, [int(cv.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError("jpeg encode failed")
    return buf.tobytes()


def decode_bytes(data: bytes) -> np.ndarray:
    img = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image bytes")
    return img


def read_image(arg) -> np.ndarray:
    """An image argument: ``{"b64": ...}`` (sent by the client) or
    ``{"server": path}`` (a file on this unit)."""
    if isinstance(arg, dict) and "b64" in arg:
        return decode_bytes(base64.b64decode(arg["b64"]))
    if isinstance(arg, dict) and "server" in arg:
        img = cv.imread(os.path.expanduser(str(arg["server"])))
        if img is None:
            raise ValueError(f"cannot read image {arg['server']!r} on the vision unit")
        return img
    raise ValueError("an image is a path, bytes or an array on the client, or {'server': path}")


def preset_key(key: str, key_path: str, base_dir: str) -> str:
    """The provider key: ``key`` when it is not empty, else the contents of
    the plain-text file ``key_path`` (relative to ``base_dir`` — the
    config file's folder — or absolute; ``~`` expands; surrounding
    whitespace ignored). Raises when there is none — never a call without
    a key."""
    if key and str(key).strip():
        return str(key).strip()
    p = os.path.expanduser(str(key_path or ""))
    if not p.strip():
        raise ValueError("give key, or key_path (a file holding just the key)")
    if not os.path.isabs(p):
        p = os.path.join(base_dir, p)
    if not os.path.isfile(p):
        raise ValueError(f"key_path {key_path!r}: no such file ({p})")
    with open(p) as fh:
        k = fh.read().strip()
    if not k:
        raise ValueError(f"key_path {key_path!r}: the file is empty")
    return k


def _validate(det: dict, name: str) -> dict:
    where = f"vlm detection {name!r}"
    raw = {k: v for k, v in det.items() if k not in ("cmd", "log_name")}
    leaked = [k for k in KEY_FIELDS if k in raw]
    if leaked:
        raise ValueError(f"{where}: {', '.join(leaked)} must be resolved before the server")
    extra = sorted(set(raw) - set(FIELDS))
    missing = [f for f in FIELDS if f not in raw]
    if extra or missing:
        hint = (" — model is a mapping: {name, backend, thinking}"
                if {"backend", "thinking"} & set(extra) else "")
        raise ValueError(f"{where}: " + "; ".join(
            ([f"unknown field(s) {', '.join(extra)}"] if extra else []) +
            ([f"missing {', '.join(missing)} (every field is written out)"] if missing else [])) + hint)
    m = raw["model"]
    if not isinstance(m, dict) or set(m) != set(MODEL_FIELDS):
        raise ValueError(f"{where}: model is {{name, backend, thinking}}")
    for k in ("name", "backend"):
        if not isinstance(m[k], str) or not m[k]:
            raise ValueError(f"{where}: model.{k} must be a name")
    if m["thinking"] not in THINKING:
        raise ValueError(f"{where}: model.thinking must be one of {', '.join(THINKING)}")
    if raw["output"] not in OUTPUTS:
        raise ValueError(f"{where}: output must be one of {', '.join(OUTPUTS)}")
    labels = raw["labels"]
    if not isinstance(labels, list) or not all(isinstance(x, str) and x for x in labels):
        raise ValueError(f"{where}: labels must be a list of names ([] = the model answers in its own words)")
    if raw["output"] == "ocr" and labels:
        raise ValueError(f"{where}: ocr reads text, there is nothing to choose from — labels must be []")
    if not isinstance(raw["schema"], dict):
        raise ValueError(f"{where}: schema must be a mapping — the JSON schema of extra data on every entry, {{}} for none")
    if not isinstance(raw["prompt"], str) or not raw["prompt"].strip():
        raise ValueError(f"{where}: prompt must be text")
    if not isinstance(raw["references"], list):
        raise ValueError(f"{where}: references must be a list ([] for none)")
    for i, r in enumerate(raw["references"]):
        if not isinstance(r, dict) or set(r) != {"image", "label", "note"}:
            raise ValueError(f"{where}: references[{i}] is {{image, label, note}}")
    if not isinstance(raw["image_size"], int) or raw["image_size"] < 64:
        raise ValueError(f"{where}: image_size must be an integer (px), at least 64")
    if not isinstance(raw["timeout_s"], (int, float)) or raw["timeout_s"] <= 0:
        raise ValueError(f"{where}: timeout_s must be a positive number")
    return raw


_CACHE: Dict[str, Preset] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 32


def load_preset(det: dict, name: str) -> Preset:
    """A final vlm ``detection`` dict -> :class:`Preset`. ``name`` is the
    detection's name."""
    raw = _validate(det, name)
    key = hashlib.sha256(json.dumps([name, raw], sort_keys=True, default=str).encode()).hexdigest()[:16]
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
    if hit is not None:
        return hit
    refs = []
    for i, r in enumerate(raw["references"]):
        try:
            img = read_image(r["image"])
        except ValueError as ex:
            raise ValueError(f"vlm detection {name!r}: references[{i}]: {ex}") from None
        refs.append(Reference(str(r["label"]), str(r["note"]), encode_image(img, raw["image_size"])))
    m = raw["model"]
    p = Preset(name=name, hash=key, model=m["name"], backend=m["backend"], thinking=m["thinking"],
               output=raw["output"], labels=list(raw["labels"]), schema=raw["schema"],
               prompt=raw["prompt"], references=refs, image_size=raw["image_size"],
               timeout_s=float(raw["timeout_s"]))
    with _CACHE_LOCK:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = p
    return p
