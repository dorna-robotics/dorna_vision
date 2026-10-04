"""VLM detections — a hosted vision-language model as a detection.

The contract is docs/vision-guide.md §9 (workspace repo). The pieces:

    preset.py    a vlm detection's settings (its ``detection`` dict, usually a
                 config file) -> Preset, validated, references decoded once,
                 cached by content hash
    backend.py   the provider boundary: Backend.answer(...) -> neutral answer,
                 and the neutral answer -> detection entries mapping
    default.py   the first backend

Nothing outside default.py knows which provider answers.
"""
from .backend import VlmError, get_backend, to_entries
from .preset import Preset, decode_bytes, encode_image, load_preset, preset_key, read_image
