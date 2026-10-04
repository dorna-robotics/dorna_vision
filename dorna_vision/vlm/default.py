"""The first VLM backend — ``backend: default`` in a preset.

Today this is Google's hosted model, through its Interactions REST
endpoint. This module is the ONLY place that knows it: the request
format, the structured-output schema, ``thinking`` -> ``thinking_level``,
the provider's box convention ([ymin, xmin, ymax, xmax] on 0..1000),
and its errors. Swapping providers is a new module beside this one.

Caching: the provider caches a repeated request PREFIX on its own
(implicit caching; from 4,096 tokens on its Flash models). The request
is built stable-part-first — prompt, then the preset's references — so
every call of one preset shares that prefix; ``tokens_cached`` in the
result shows whether it hit.
"""
from __future__ import annotations

import base64
import json
import threading

import requests

from .backend import Backend, VlmError, register

URL = "https://generativelanguage.googleapis.com/v1beta/interactions"

# A config's ``model.name`` -> this provider's model id. Explicit: an
# unknown name is an error listing these, never a guess.
MODELS = {
    "flash-3.8": "gemini-3.8-flash",
    "flash-3.7": "gemini-3.7-flash",
    "flash-3.6": "gemini-3.6-flash",
    "flash-lite-3.5": "gemini-3.5-flash-lite",
    "pro-3.1": "gemini-3.1-pro-preview",
}
MAX_OUTPUT_TOKENS = 4096

_local = threading.local()


def _session() -> requests.Session:
    """One keep-alive connection per worker thread — the TLS handshake is
    paid once, not per call."""
    s = getattr(_local, "s", None)
    if s is None:
        s = _local.s = requests.Session()
    return s


def _text(t):
    return {"type": "text", "text": t}


def _image(jpeg: bytes):
    return {"type": "image", "mime_type": "image/jpeg",
            "data": base64.b64encode(jpeg).decode()}


_CONF = {"type": "number", "minimum": 0, "maximum": 1}
_REASON = {"type": "string", "description": "one short sentence"}
_BOX = {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000},
        "minItems": 4, "maxItems": 4,
        "description": "[ymin, xmin, ymax, xmax] normalised to 0-1000 on that view's image"}


def _schema(preset, n_views: int) -> dict:
    """The structured-output schema: the fixed envelope for ``output``,
    labels as an enum when given, and the config's ``schema`` as ``data``
    on every entry when it is not empty."""
    label = {"type": "string", "enum": list(preset.labels)} if preset.labels else {"type": "string"}
    view = {"type": "integer", "minimum": 1, "maximum": n_views}
    data = {"data": preset.schema} if preset.schema else {}
    if preset.output == "cls":
        props = {"label": label, "confidence": _CONF, **data, "reason": _REASON}
    else:
        head = ({"label": label} if preset.output == "od" else {"text": {"type": "string"}})
        item_props = {**head, "view": view, "box_2d": _BOX, "confidence": _CONF, **data}
        item = {"type": "object", "required": list(item_props), "properties": item_props}
        props = {("objects" if preset.output == "od" else "lines"): {"type": "array", "items": item},
                 "reason": _REASON}
    return {"type": "object", "properties": props, "required": list(props)}


def _instruction(preset, n: int) -> str:
    one = "All views show ONE part; answer for that part, using every view. " if n > 1 else ""
    data = " data filled exactly to its schema," if preset.schema else ""
    box = ("view (its view number), box_2d as [ymin, xmin, ymax, xmax] normalised to 0-1000 "
           "on that view's image,")
    if preset.output == "cls":
        lab = (f"exactly one of: {', '.join(preset.labels)}" if preset.labels
               else "a short answer, a few words")
        return (f"{one}Give label ({lab}),{data} confidence from 0 to 1, "
                "and reason in one short sentence.")
    if preset.output == "od":
        what = (f"every object of these kinds: {', '.join(preset.labels)}" if preset.labels
                else "every distinct object, naming each in a word or two as its label")
        return (f"Find {what}, in every view. For each give label, {box}{data} and confidence "
                "from 0 to 1. Give reason in one short sentence.")
    return (f"Read every line of text in every view. For each give text, {box}{data} and "
            "confidence from 0 to 1. Give reason in one short sentence.")


def _neutral(preset, ans: dict) -> dict:
    """Provider answer -> the neutral answer (backend.py): box_2d on
    0..1000 [ymin, xmin, ymax, xmax] -> box on 0..1 [x0, y0, x1, y1]."""
    def box(b):
        if not (isinstance(b, list) and len(b) == 4):
            raise VlmError(f"box_2d must be 4 numbers, got {b!r}")
        ymin, xmin, ymax, xmax = (float(v) / 1000.0 for v in b)
        return [xmin, ymin, xmax, ymax]
    if preset.output in ("od", "ocr"):
        key = "objects" if preset.output == "od" else "lines"
        items = ans.get(key)
        if not isinstance(items, list):
            raise VlmError(f"answer has no {key} list")
        out = []
        for it in items:
            it = dict(it)
            it["box"] = box(it.pop("box_2d", None))
            out.append(it)
        return {key: out, "reason": ans.get("reason", "")}
    return ans


class DefaultBackend(Backend):
    name = "default"

    def answer(self, preset, views, extra_refs, extra_prompt, key):
        if preset.model not in MODELS:
            raise VlmError(f"backend default has no model.name {preset.model!r} "
                           f"(known: {', '.join(MODELS)})")
        if not key:
            raise VlmError("no vlm_key")
        n = len(views)
        parts = [_text(preset.prompt)]
        i = 0
        for r in preset.references:                       # ── the stable prefix
            i += 1
            parts += [_text(f"Reference {i} — {r.label}: {r.note}"), _image(r.jpeg)]
        for label, note, jpeg in extra_refs:              # ── per call from here
            i += 1
            parts += [_text(f"Reference {i} — {label}: {note}"), _image(jpeg)]
        if extra_prompt:
            parts.append(_text(extra_prompt))
        for k, jpeg in enumerate(views, 1):
            parts += [_text(f"Part to judge — view {k} of {n}"), _image(jpeg)]
        parts.append(_text(_instruction(preset, n)))
        body = {
            "model": MODELS[preset.model],
            "input": parts,
            "generation_config": {"thinking_level": preset.thinking,
                                  "max_output_tokens": MAX_OUTPUT_TOKENS},
            "response_format": {"type": "text", "mime_type": "application/json",
                                "schema": _schema(preset, n)},
        }
        try:
            r = _session().post(URL, json=body, timeout=preset.timeout_s,
                                headers={"x-goog-api-key": key})
        except requests.Timeout:
            raise VlmError(f"backend timed out after {preset.timeout_s:g} s") from None
        except requests.RequestException as ex:
            raise VlmError(f"backend unreachable: {type(ex).__name__}") from None
        try:
            data = r.json()
        except ValueError:
            raise VlmError(f"backend HTTP {r.status_code}: not JSON") from None
        if r.status_code != 200:
            msg = (data.get("error") or {}).get("message") if isinstance(data, dict) else None
            raise VlmError(f"backend HTTP {r.status_code}: {msg or 'error'}")
        text = data.get("output_text")
        if not text:
            text = "".join(c.get("text", "")
                           for s in data.get("steps", []) if s.get("type") == "model_output"
                           for c in s.get("content", []) if c.get("type") == "text")
        if not text:
            raise VlmError(f"backend gave no answer (status {data.get('status', '?')})")
        try:
            ans = json.loads(text)
        except ValueError:
            raise VlmError("backend answer is not JSON") from None
        u = data.get("usage") or {}
        return {"answer": _neutral(preset, ans), "model": data.get("model") or MODELS[preset.model],
                "usage": {"tokens_in": u.get("total_input_tokens"),
                          "tokens_cached": u.get("total_cached_tokens"),
                          "tokens_out": u.get("total_output_tokens"),
                          "tokens_thought": u.get("total_thought_tokens")}}


register(DefaultBackend())
