"""
Command handlers.

Each handler is a plain sync function: handler(session, args) -> result.

Return value conventions:
  - dict / list / scalar -> JSON reply payload under {"id": id, **result_wrapping}
  - a tuple (json_dict, binary_bytes) -> JSON reply followed by a binary WS frame

Handlers never touch the websocket directly. The ws_handler dispatches them
onto the right executor and sends replies.

Commands that touch a Detection's state or grab camera frames are routed onto
the camera's single-worker executor (see CAMERA_BOUND in ws_handler). That
executor is resolved from the detection's camera serial number.
"""
import cv2 as cv


# -------------------- hello --------------------

def hello(session, args):
    from dorna_vision import __version__
    return {"server": "dorna_vision", "version": __version__, "protocol": 1}


# -------------------- camera --------------------

def camera_list(session, args):
    """
    Return all cameras the GUI cares about: attached USB devices plus any
    pool entries (which may include unplugged-but-still-pooled cameras).
    Each item carries:
      attached: bool   True if it is currently visible on the USB bus
      added:    bool   True if it is currently in the server's pool
    """
    hardware = {d["serial_number"]: d for d in session.camera_pool.list_devices()}
    pooled = set(session.camera_pool.list_pool_keys())

    out = []
    for sn, dev in hardware.items():
        out.append({**dev, "attached": True, "added": sn in pooled})
    for sn in pooled - set(hardware.keys()):
        out.append({"serial_number": sn, "attached": False, "added": True})
    return {"devices": out}


def camera_add(session, args):
    args = dict(args)
    serial_number = args.get("serial_number")
    if not serial_number:
        raise ValueError("serial_number is required")
    session.camera_add(**args)
    return {"serial_number": serial_number}


def camera_remove(session, args):
    serial_number = args.get("serial_number")
    if not serial_number:
        raise ValueError("serial_number is required")
    ok = session.camera_remove(serial_number)
    return {"serial_number": serial_number, "removed": ok}


def camera_get_img(session, args):
    """
    Grab a fresh frame directly from a pooled Camera and return it as JPEG.
    type: "color_img" (default) | "depth_img" | "ir_img"
    """
    sn = args.get("serial_number")
    if not sn:
        raise ValueError("serial_number is required")
    cam = session.camera_pool.get(sn)
    if cam is None:
        raise ValueError("camera not found: %s" % sn)

    kind = args.get("type", "color_img")
    quality = int(args.get("quality", 75))

    _df, _irf, _cf, depth_img, ir_img, color_img, _di, _frames, _ts = cam.get_all()

    if kind == "color_img":
        img = color_img
    elif kind == "depth_img":
        img = depth_img
    elif kind == "ir_img":
        img = ir_img
    else:
        raise ValueError("unknown image type: %s" % kind)

    if img is None:
        raise ValueError("no image available for type %s" % kind)

    ok, buf = cv.imencode(".jpg", img, [int(cv.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    meta = {"serial_number": sn, "type": kind, "shape": list(img.shape), "encoding": "jpeg", "quality": quality}
    return (meta, buf.tobytes())


# -------------------- robot --------------------

def robot_add(session, args):
    host = args.get("host")
    if not host:
        raise ValueError("host is required")
    port = args.get("port", 443)
    timeout = args.get("timeout", 5)
    model = args.get("model", "dorna_ta")
    config = args.get("config", None)
    session.robot_add(host=host, port=port, timeout=timeout, model=model, config=config)
    return {"host": host}


def robot_remove(session, args):
    host = args.get("host")
    if not host:
        raise ValueError("host is required")
    ok = session.robot_remove(host)
    return {"host": host, "removed": ok}


# -------------------- detection --------------------

def detection_add(session, args, data=None):
    """
    Create a Detection in this client's session.

    If `data` is provided (i.e. the client sent a binary frame after the
    envelope), it's the ML model weights. We splat them into a per-call
    temp file just long enough for Detection.__init__ to load them into
    memory, then delete it — no model file lingers on disk past this call.
    """
    import os
    import tempfile

    args = dict(args)
    name = args.pop("name", None)
    if not name:
        raise ValueError("name is required")
    camera_serial_number = args.pop("camera_serial_number", None)
    robot_host = args.pop("robot_host", None)

    tmp_path = None
    if data:
        det_pkg = dict(args.get("detection") or {})
        # Pick a sensible suffix from the existing path hint, else .pkl
        suffix = ".pkl"
        hint = det_pkg.get("path") or ""
        ext = os.path.splitext(str(hint))[1]
        if ext:
            suffix = ext
        fd, tmp_path = tempfile.mkstemp(prefix="dorna_vision_model_", suffix=suffix)
        os.close(fd)
        with open(tmp_path, "wb") as f:
            f.write(data)
        det_pkg["path"] = tmp_path

        # Auto-detect the detection cmd from the pickle's meta.type when
        # the caller didn't pre-set one. This lets the GUI just hand us a
        # model file without asking the user "is this OD or ROD or KP…?"
        # — the answer is already baked into the pickle by training.
        if "cmd" not in det_pkg:
            try:
                import pickle as _p
                with open(tmp_path, "rb") as _f:
                    _model_dict = _p.load(_f)
                _t = ((_model_dict.get("meta") or {}).get("type") or "").lower()
                if _t in ("od", "rod", "cls", "kp", "anom"):
                    det_pkg["cmd"] = _t
            except Exception:
                pass

        args["detection"] = det_pkg

    try:
        session.detection_add(
            name=name,
            camera_serial_number=camera_serial_number,
            robot_host=robot_host,
            **args,
        )
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # Echo the detected/effective cmd back so the GUI can filter the
    # runtime method picker without having to ask the user up front.
    cmd = (args.get("detection") or {}).get("cmd")
    return {"name": name, "cmd": cmd}


def detection_run(session, args):
    args = dict(args)
    name = args.pop("name", None)
    if not name:
        raise ValueError("name is required")
    use_last = args.pop("use_last", False)
    det = session.detection_get(name)

    data = args.pop("data", None)
    if use_last and det.camera_data is not None:
        data = det.camera_data

    valid = det.run(data=data, **args)
    return {"name": name, "valid": _to_jsonable(valid)}


def detection_get_img(session, args):
    name = args.get("name")
    if not name:
        raise ValueError("name is required")
    kind = args.get("type", "img")
    quality = int(args.get("quality", 85))
    det = session.detection_get(name)

    cam_data = det.retval.get("camera_data") if det.retval else None
    img = None
    if kind == "img":
        img = getattr(det, "img", None) if cam_data is None else cam_data.get("img", getattr(det, "img", None))
    elif kind == "img_roi":
        img = cam_data.get("img_roi") if cam_data else None
    elif kind == "img_thr":
        img = getattr(det, "img_thr", None)
    elif kind in ("color_img", "depth_img", "ir_img"):
        img = cam_data.get(kind) if cam_data else None
    else:
        raise ValueError("unknown image type: %s" % kind)

    if img is None:
        raise ValueError("no image available for type %s" % kind)

    ok, buf = cv.imencode(".jpg", img, [int(cv.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    meta = {"name": name, "type": kind, "shape": list(img.shape), "encoding": "jpeg", "quality": quality}
    return (meta, buf.tobytes())


def detection_xyz(session, args):
    name = args.get("name")
    pxl = args.get("pxl")
    if not name or pxl is None:
        raise ValueError("name and pxl are required")
    det = session.detection_get(name)
    return {"name": name, "xyz": _to_jsonable(det.xyz(pxl))}


def detection_pixel(session, args):
    name = args.get("name")
    xyz = args.get("xyz")
    if not name or xyz is None:
        raise ValueError("name and xyz are required")
    det = session.detection_get(name)
    return {"name": name, "pxl": _to_jsonable(det.pixel(xyz))}


def detection_grasp(session, args):
    from dorna_vision import grasp as grasp_mod

    args = dict(args)
    name = args.pop("name", None)
    if not name:
        raise ValueError("name is required")
    det = session.detection_get(name)

    required = ("target_id", "target_rvec", "gripper_opening", "finger_wdith", "finger_location")
    for k in required:
        if k not in args:
            raise ValueError("missing arg: %s" % k)

    best_rvec = grasp_mod.collision_free_rvec(
        target_id=args["target_id"],
        target_rvec=args["target_rvec"],
        gripper_opening=args["gripper_opening"],
        finger_wdith=args["finger_wdith"],
        finger_location=args["finger_location"],
        detection_obj=det,
        mask_type=args.get("mask_type", "bb"),
        prune_factor=args.get("prune_factor", 2),
        num_steps=args.get("num_steps", 360),
        search_angle=tuple(args.get("search_angle", (0, 360))),
    )
    return {"name": name, "rvec": _to_jsonable(best_rvec)}


def detection_remove(session, args):
    name = args.get("name")
    if not name:
        raise ValueError("name is required")
    ok = session.detection_remove(name)
    return {"name": name, "removed": ok}


def detection_list(session, args):
    return {"detections": session.detection_list()}


# -------------------- dynamic RPC --------------------

def call(session, args, data=None):
    """
    Generic RPC: call any method on a pooled Detection / Camera / Dorna.

    If `data` is present (the client sent a binary frame after the
    envelope), it gets spliced into kwargs as the `data` keyword. The
    canonical use is `vc.detection(name).run(data=<image bytes>)` — the
    bytes flow straight through cv.imdecode in Detection.run, no temp
    file on the server.

    {"cmd": "call", "args": {
        "target":  "detection" | "camera" | "robot",
        "name":    <detection name | camera serial_number | robot name>,
        "method":  "<method-name-on-the-underlying-object>",
        "args":    [...],      # positional args (optional)
        "kwargs":  {...}       # keyword args  (optional)
    }}
    """
    target = args.get("target")
    name = args.get("name")
    method = args.get("method")
    call_args = args.get("args", []) or []
    call_kwargs = dict(args.get("kwargs", {}) or {})

    if data is not None:
        # The binary frame is encoded image bytes (typically a JPEG/PNG
        # for detection.run). Decode here and feed it through the
        # "keep current camera_data" branch in Detection.get_camera_data
        # so the run uses the bytes directly — no temp file, no disk.
        import time
        import numpy as np
        import cv2 as cv

        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv.imdecode(arr, cv.IMREAD_COLOR)
        if img is None:
            raise ValueError("could not decode image bytes")
        feed = "color_img"
        if target == "detection":
            try:
                det = session.detection_get(name)
                feed = getattr(det, "feed", "color_img") or "color_img"
            except Exception:
                pass
        call_kwargs["data"] = {feed: img, "timestamp": time.time()}

    if not target or not name or not method:
        raise ValueError("target, name, method are required")
    if method.startswith("_"):
        raise ValueError("private names are not exposed")

    obj = _resolve_target(session, target, name)
    if not hasattr(obj, method):
        raise ValueError("%s has no attribute: %s" % (target, method))

    attr = getattr(obj, method)
    if callable(attr):
        result = attr(*list(call_args), **call_kwargs)
        kind = "method"
    else:
        # Plain attribute access — args/kwargs are ignored
        result = attr
        kind = "attribute"

    return {"target": target, "name": name, "method": method, "kind": kind,
            "result": _to_jsonable(result)}


def _resolve_target(session, target, name):
    if target == "detection":
        return session.detection_get(name)
    if target == "camera":
        cam = session.camera_pool.get(name)
        if cam is None:
            raise ValueError("camera not found: %s" % name)
        return cam
    if target == "robot":
        robot = session.robot_pool.get(name)
        if robot is None:
            # Lazy acquire with defaults; explicit robot_add can override.
            robot = session.robot_pool.acquire(host=name)
        return robot
    raise ValueError("unknown target: %s" % target)


# -------------------- dispatch table --------------------

HANDLERS = {
    "hello": hello,
    "camera_list": camera_list,
    "camera_add": camera_add,
    "camera_remove": camera_remove,
    "camera_get_img": camera_get_img,
    "robot_add": robot_add,
    "robot_remove": robot_remove,
    "detection_add": detection_add,
    "detection_list": detection_list,
    "detection_run": detection_run,
    "detection_get_img": detection_get_img,
    "detection_xyz": detection_xyz,
    "detection_pixel": detection_pixel,
    "detection_grasp": detection_grasp,
    "detection_remove": detection_remove,
    "call": call,
}

# Commands that MAY take a binary frame from the client immediately after
# their JSON envelope (when the envelope sets binary_follows=true). The
# handler signature for these is (session, args, data) where data is bytes
# or None. Used to ship images and ML model weights inline rather than
# staging them on the server's filesystem.
BINARY_INBOUND = {
    "detection_add",   # binary = ML model bytes (loaded into Detection then discarded)
    "call",            # binary = the 'data' kwarg for whatever method is being called (e.g. detection.run)
}

# Commands that must run on the camera's single-worker executor so they
# serialize against detection_run. The detection name in args resolves to a
# camera serial number via session.detection_camera_serial_number().
CAMERA_BOUND = {
    "detection_run",
    "detection_get_img",
    "detection_xyz",
    "detection_pixel",
    "detection_grasp",
    "camera_get_img",
}


# -------------------- json helpers --------------------

# Numpy arrays larger than this get replaced with a short placeholder dict.
# Keeps the full retval reachable without dumping ~1 MB images into JSON.
# 1024 elements comfortably covers all real intrinsics / calibration / detection
# metadata (3x3, 4x4, Nx2 corners, etc.) and shortens anything image-sized.
_MAX_ARRAY_ELEMENTS = 1024


def _to_jsonable(obj):
    """
    Convert anything to a JSON-safe value.
      * numpy arrays: small -> .tolist(); large -> shape/dtype placeholder
      * numpy scalars -> Python scalar
      * dict / list / tuple: recurse
      * plain scalars / str / None: passthrough
      * anything else: placeholder dict with the type name
    """
    import numpy as np

    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        if obj.size <= _MAX_ARRAY_ELEMENTS:
            return obj.tolist()
        return {
            "_placeholder": "ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "size": int(obj.size),
        }
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    # Unknown type (pyrealsense2 frames, custom classes, etc.) — keep the
    # field reachable but mark it so the client can see what was there.
    return {"_placeholder": "nonserializable", "type": type(obj).__name__}
