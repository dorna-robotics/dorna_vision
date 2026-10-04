"""Detection config files — vision-guide §9 "Config files".

A config file holds any of a detection's settings, with the same keys
and nesting ``detection_add`` takes::

    detection: {cmd: od, path: tube.pkl, conf: 0.5}
    roi: {corners: [[100, 50], [700, 600]], inv: 0, crop: 1}
    display: {label: 1, client_save_img: "tube_od/"}

ONE rule for every path in it: a relative path is relative to the file
it is written in (or it is absolute; ``~`` expands): ``detection.path``
(a model), ``detection.references[].image`` (vlm), ``detection.key_path``
(vlm). Keys given inline win over the file, key by key at every depth; a
list is replaced whole.

The client applies this to a config on its own computer; this module
applies it to ``config: {"server": path}`` — a file on this unit. The
two implement the same rule (the client package cannot import this one).
One consequence for a file on THIS unit: its ``display.client_save_*``
values name folders on the CALLING computer, which a file here cannot
reach relatively — they must be absolute (or ``false``); a relative one,
``true`` included, is refused.
"""
from __future__ import annotations

import copy
import os

import yaml


def deep_merge(base: dict, over: dict) -> dict:
    """``over`` wins, key by key at every depth; non-dicts (lists
    included) are replaced whole. Neither input is modified."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _abs(p: str, base_dir: str) -> str:
    p = os.path.expanduser(str(p))
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))


def load_config(path: str, server_images: bool = True) -> dict:
    """Read a config file and make its file paths absolute (against the
    file's folder). With ``server_images`` (a file on this unit) a vlm
    reference image becomes ``{"server": abs path}``."""
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(path):
        raise ValueError(f"config {path!r}: no such file")
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path!r}: must be a mapping of a detection's settings")
    base = os.path.dirname(path)
    disp = cfg.get("display")
    if isinstance(disp, dict):
        for k in ("client_save_img", "client_save_img_roi", "client_save_log"):
            v = disp.get(k)
            rel = (v is True or v == 1) or (isinstance(v, str) and v.strip()
                                             and not os.path.isabs(os.path.expanduser(v)))
            if rel:
                raise ValueError(f"config {path!r}: display.{k} {v!r} names a folder on the "
                                 f"CALLING computer, which a file on the vision unit cannot "
                                 f"name relatively — write an absolute path there, or false")
    det = cfg.get("detection")
    if isinstance(det, dict):
        if isinstance(det.get("path"), str) and det["path"]:
            det["path"] = _abs(det["path"], base)
        if isinstance(det.get("key_path"), str) and det["key_path"].strip():
            det["key_path"] = _abs(det["key_path"], base)
        for r in det.get("references") or []:
            if isinstance(r, dict) and isinstance(r.get("image"), str):
                r["image"] = {"server": _abs(r["image"], base)} if server_images else _abs(r["image"], base)
    return cfg
