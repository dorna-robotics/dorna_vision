"""
ML inference for the dorna_vision pipeline, on two engines — one code
base for the Pi and the Jetson:

    openvino   the Pi's CPU      ai/ov.py    OD, ROD, KP (top-down), CLS, ANOM, OCR
    tensorrt   the Jetson's GPU  ai/trt.py   OD, SEG, KP, CLS, OCR

A model file says which engine runs it — an OpenVINO model carries its
IR ('xml' / 'bin'), a TensorRT one its ONNX graph ('onnx') — so it is
never guessed, and a model the unit cannot run is refused with what it
is. A unit has one engine installed (the package's `pi` or `jetson`
extra); only the engine a model needs is imported.

Every type returns the same shape on either engine: OD / SEG / ROD / KP a
list of objects (prob, cls, rect, color; SEG + polygon; ROD + corners; KP
+ keypoints), CLS a list of [name, prob, color], ANOM one verdict, OCR
text lines. A KP model may be top-down (`top_down`: it looks in a box it
is given, one object) or find its objects itself.

    prepare(path)   once per new model on a unit: TensorRT builds its
                    engine for this GPU (minutes); OpenVINO has nothing
                    to prepare (it compiles as a model loads)
    fetch_ocr()     once per unit: the TensorRT OCR models (downloaded and
                    prepared); OpenVINO's are built in
"""

import importlib
import importlib.util

from dorna_vision.ai.common import ModelNotPreparedError, _color_for_class, _load_pickle


# The model types each engine runs (meta["type"] / a detection's cmd).
MODEL_TYPES = {
    "openvino": ("od", "rod", "kp", "cls", "anom"),
    "tensorrt": ("od", "seg", "kp", "cls"),
}
ML_CMDS = tuple(sorted({t for types in MODEL_TYPES.values() for t in types}))

_MODULES = {"openvino": "dorna_vision.ai.ov", "tensorrt": "dorna_vision.ai.trt"}
_RUNTIMES = {"openvino": "openvino", "tensorrt": "tensorrt"}   # the python package each needs
_NAMES = {"openvino": "OpenVINO (the Pi's engine)", "tensorrt": "TensorRT (the Jetson's engine)"}


def model_engine(data):
    """The engine a model (a loaded pickle) runs on."""
    if "onnx" in data:
        return "tensorrt"
    if "xml" in data and "bin" in data:
        return "openvino"
    raise ValueError("Not a model this vision server runs: no ONNX graph ('onnx', TensorRT) and no "
                     "OpenVINO IR ('xml' / 'bin') — export it with its training notebook.")


def model_info(path):
    """A model file's {"type", "engine"}."""
    data = _load_pickle(path)
    return {"type": ((data.get("meta") or {}).get("type") or "").lower(), "engine": model_engine(data)}


def unit_engine():
    """The engine installed on this unit — exactly one."""
    installed = [e for e, package in _RUNTIMES.items() if importlib.util.find_spec(package) is not None]
    if len(installed) != 1:
        raise RuntimeError("This unit has %s installed; it needs exactly one engine — install the "
                           "package's `pi` (OpenVINO) or `jetson` (TensorRT) extra."
                           % (" and ".join(installed) or "no ML engine"))
    return installed[0]


def _engine(engine):
    """The engine's module — imported only when a model needs it."""
    if importlib.util.find_spec(_RUNTIMES[engine]) is None:
        raise ValueError("This model runs on %s; this unit has no %s — it runs models for %s."
                         % (_NAMES[engine], _RUNTIMES[engine], _NAMES[unit_engine()]))
    return importlib.import_module(_MODULES[engine])


def _load(cmd, path, **kwargs):
    data = _load_pickle(path)
    engine = model_engine(data)
    if cmd not in MODEL_TYPES[engine]:
        raise ValueError("%s is not a model type the %s engine runs (it runs %s)."
                         % (cmd, engine, ", ".join(MODEL_TYPES[engine])))
    model = getattr(_engine(engine), cmd.upper())(data, **kwargs)
    model.engine = engine
    return model


def OD(path, **kwargs):   return _load("od", path, **kwargs)
def SEG(path, **kwargs):  return _load("seg", path, **kwargs)
def ROD(path, **kwargs):  return _load("rod", path, **kwargs)
def KP(path, **kwargs):   return _load("kp", path, **kwargs)
def CLS(path, **kwargs):  return _load("cls", path, **kwargs)
def ANOM(path, **kwargs): return _load("anom", path, **kwargs)


def OCR(**kwargs):
    """The unit's built-in OCR."""
    return _engine(unit_engine()).OCR(**kwargs)


def prepare(path):
    """
    Make a model ready to load on this unit — once per new .pkl. Returns
    {"built": bool, "seconds": float}: built False when there was nothing
    to build (already prepared, or an OpenVINO model).
    """
    data = _load_pickle(path)
    engine = model_engine(data)
    if engine == "openvino":
        _engine(engine)
        return {"built": False, "seconds": 0.0}
    return _engine(engine).prepare(data)


def fetch_ocr():
    """Once per unit: the OCR models, ready. {part: prepare()'s result};
    {} on OpenVINO (its models ship with the package)."""
    engine = unit_engine()
    return _engine(engine).fetch_ocr() if engine == "tensorrt" else {}


__all__ = ["MODEL_TYPES", "ML_CMDS", "ModelNotPreparedError", "model_engine", "model_info", "unit_engine",
           "OD", "SEG", "ROD", "KP", "CLS", "ANOM", "OCR", "prepare", "fetch_ocr"]
