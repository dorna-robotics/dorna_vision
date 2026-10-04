"""The provider boundary.

A backend turns (preset, views, per-call extras, key) into a NEUTRAL
answer. Everything provider-specific — the request format, the
provider's structured-output schema, how ``thinking`` maps, its box
convention, its caching, the HTTP call and its errors — lives inside
one backend module. Nothing else (the Detection, the server, the
client, the workspace, a preset) knows which provider answered.

THE NEUTRAL ANSWER, by ``preset.output``:

    cls   {"label": str, "confidence": 0..1, "data": {...}, "reason": str}
    od    {"objects": [{"label", "confidence", "view": 1..N,
                        "box": [x0, y0, x1, y1], "data": {...}}], "reason": str}
    ocr   {"lines":   [{"text",  "confidence", "view": 1..N,
                        "box": [x0, y0, x1, y1], "data": {...}}], "reason": str}

``box`` is normalised to 0..1 on that view's image (x right, y down).
``data`` is the config's ``schema`` filled in; absent or {} when the
schema is {}. ``label`` is one of the config's labels, or the model's own
words when they are [].
``answer()`` returns ``{"answer": <neutral>, "model": <provider model
id>, "usage": {"tokens_in", "tokens_cached", "tokens_out"}}`` and raises
:class:`VlmError` on anything that is not a valid answer — never a guess.

``to_entries`` maps a neutral answer onto the result shape every
detection returns, so callers cannot tell a VLM from a local model.
"""
from __future__ import annotations

from typing import Any, Dict, List

BACKENDS: Dict[str, Any] = {}


class VlmError(RuntimeError):
    """The model did not produce a usable answer: network, provider
    error, timeout, refusal, or an answer outside the config's labels.
    Raised — the workflow pauses like a camera failure."""


class Backend:
    """Subclass, set ``name``, implement ``answer``, then ``register``."""
    name = ""

    def answer(self, preset, views: List[bytes], extra_refs: List[tuple],
               extra_prompt: str, key: str) -> Dict[str, Any]:
        """``views``: JPEG bytes, already at ``preset.image_size``.
        ``extra_refs``: ``(label, note, jpeg)`` per call."""
        raise NotImplementedError


def register(backend: Backend) -> Backend:
    BACKENDS[backend.name] = backend
    return backend


def get_backend(name: str) -> Backend:
    if name not in BACKENDS:
        # the first backend registers itself on import
        from . import default  # noqa: F401
    if name not in BACKENDS:
        raise VlmError(f"no vlm backend {name!r} (known: {', '.join(sorted(BACKENDS))})")
    return BACKENDS[name]


def _conf(v) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def _box_to_px(box, view) -> list:
    """Normalised [x0, y0, x1, y1] on a view -> its 4 corners in FRAME
    pixels (the view's crop offset added back)."""
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        raise VlmError(f"box must be [x0, y0, x1, y1], got {box!r}")
    h, w = view["shape"]
    ox, oy = view["offset"]
    x0, y0, x1, y1 = (_conf(v) for v in box)
    x0, x1 = sorted((x0 * w + ox, x1 * w + ox))
    y0, y1 = sorted((y0 * h + oy, y1 * h + oy))
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def to_entries(preset, answer: Dict[str, Any], views: List[Dict[str, Any]],
               timestamp) -> List[Dict[str, Any]]:
    """Neutral answer -> detection entries. ``views``: per view sent,
    ``{"shape": (h, w), "offset": (x, y)}`` — the crop's size and where it
    sits in its frame. The LAST view is the frame the run captured."""
    if not isinstance(answer, dict):
        raise VlmError("answer is not an object")
    reason = str(answer.get("reason", ""))
    out = preset.output

    def data(d):
        return d if isinstance(d, dict) else {}
    if out == "cls":
        label = answer.get("label")
        if preset.labels and label not in preset.labels:
            raise VlmError(f"answer {label!r} is not one of the config's labels {preset.labels}")
        last = views[-1]
        h, w = last["shape"]
        ox, oy = last["offset"]
        x1, y1 = ox + w - 1, oy + h - 1          # the crop's last pixel
        return [{"timestamp": timestamp, "cls": label, "conf": _conf(answer.get("confidence")),
                 "center": [(ox + x1) / 2.0, (oy + y1) / 2.0],
                 "corners": [[ox, oy], [x1, oy], [x1, y1], [ox, y1]],
                 "reason": reason, "data": data(answer.get("data"))}]
    key = "objects" if out == "od" else "lines"
    items = answer.get(key)
    if not isinstance(items, list):
        raise VlmError(f"answer has no {key} list")
    entries = []
    for it in items:
        v = it.get("view")
        if not isinstance(v, int) or not 1 <= v <= len(views):
            raise VlmError(f"view {v!r} is not 1..{len(views)}")
        corners = _box_to_px(it.get("box"), views[v - 1])
        cls = it.get("label") if out == "od" else it.get("text")
        if out == "od" and preset.labels and cls not in preset.labels:
            raise VlmError(f"object {cls!r} is not one of the config's labels {preset.labels}")
        entries.append({"timestamp": timestamp, "cls": cls, "conf": _conf(it.get("confidence")),
                        "center": [(corners[0][0] + corners[2][0]) / 2.0,
                                   (corners[0][1] + corners[2][1]) / 2.0],
                        "corners": corners, "view": v, "reason": reason, "data": data(it.get("data"))})
    return entries
