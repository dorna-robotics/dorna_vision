"""
Python client for the dorna_vision server.

Lightweight — depends only on `websocket-client`. Install with:

    pip install dorna_vision_client

Usage:

    from dorna_vision_client import VisionClient

    vc = VisionClient()
    vc.connect()                              # defaults to 127.0.0.1:8765

    vc.camera_add(serial_number="12345", mode="bgrd", stream={"width":848,"height":480,"fps":15})
    vc.robot_add(host="192.168.1.50")         # optional; robots keyed by host

    vc.detection_add(name="aruco1",
                     camera_serial_number="12345",
                     robot_host="192.168.1.50",
                     detection={"cmd":"aruco", "marker_length":20, "dictionary":"DICT_4X4_50"})

    valid = vc.detection("aruco1").run()
    jpeg, meta = vc.detection("aruco1").get_img()

    vc.close()
"""
import itertools
import json
import os
import threading

import websocket  # websocket-client


__version__ = "0.1.0"

DEFAULT_TIMEOUT = 10.0


class VisionServerError(Exception):
    def __init__(self, code, msg):
        super(VisionServerError, self).__init__("%s: %s" % (code, msg))
        self.code = code
        self.msg = msg


class _Pending(object):
    __slots__ = ("event", "json", "binary", "needs_binary", "error")

    def __init__(self):
        self.event = threading.Event()
        self.json = None
        self.binary = None
        self.needs_binary = False
        self.error = None


class VisionClient(object):
    """
    Sync client. All per-command methods block until the server replies or
    `timeout` seconds elapse.
    """

    def __init__(self):
        self._ws = None
        self._reader = None
        self._send_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending = {}              # id -> _Pending
        self._last_binary_holder = None # _Pending waiting for its binary frame
        self._id_counter = itertools.count(1)
        self._connected = False
        self._close_evt = threading.Event()
        self._default_timeout = DEFAULT_TIMEOUT
        self._last_error = None         # last reason the connection dropped

    # ---------------- connection ----------------

    def is_connected(self):
        return self._connected and self._ws is not None

    def last_error(self):
        return self._last_error

    def connect(self, host="127.0.0.1", port=8765, path="/ws", timeout=5, default_timeout=DEFAULT_TIMEOUT):
        # idempotent: drop any prior session cleanly
        if self._ws is not None or self._reader is not None or self._connected:
            self.close()

        self._default_timeout = default_timeout
        self._last_error = None
        url = "ws://%s:%d%s" % (host, port, path)

        try:
            ws = websocket.create_connection(url, timeout=timeout)
        except Exception as ex:
            raise ConnectionError("could not connect to %s: %s" % (url, ex))

        # CRITICAL: reset socket timeout to blocking-forever so the reader
        # thread's recv() does not time out on idle periods. The `timeout`
        # arg above only needs to bound the initial connect/handshake.
        try:
            ws.settimeout(None)
        except Exception:
            pass

        self._ws = ws
        self._connected = True
        self._close_evt.clear()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        try:
            self.hello(timeout=timeout)
        except Exception:
            self.close()
            raise
        return True

    def close(self, timeout=2):
        self._connected = False
        self._close_evt.set()
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass
        if self._reader is not None and self._reader is not threading.current_thread():
            self._reader.join(timeout=timeout)
        self._ws = None
        self._reader = None
        with self._pending_lock:
            for p in self._pending.values():
                p.error = ConnectionError(self._last_error or "connection closed")
                p.event.set()
            self._pending.clear()
            self._last_binary_holder = None

    # ---------------- read loop ----------------

    def _read_loop(self):
        reason = None
        while self._connected:
            try:
                frame = self._ws.recv()
            except Exception as ex:
                reason = "recv failed: %s" % ex
                break
            if frame is None or frame == "":
                reason = "server closed the connection"
                break

            if isinstance(frame, (bytes, bytearray)):
                self._on_binary(bytes(frame))
            else:
                self._on_text(frame)

        self._connected = False
        self._last_error = reason
        self._close_evt.set()
        with self._pending_lock:
            for p in self._pending.values():
                if not p.event.is_set():
                    p.error = ConnectionError(reason or "connection closed")
                    p.event.set()

    def _on_text(self, text):
        try:
            payload = json.loads(text)
        except Exception:
            return

        msg_id = payload.get("id")
        with self._pending_lock:
            pending = self._pending.get(msg_id) if msg_id is not None else None
            if pending is None:
                return

            if payload.get("binary_follows"):
                pending.json = payload
                pending.needs_binary = True
                self._last_binary_holder = pending
                return

            pending.json = payload
            pending.event.set()
            self._pending.pop(msg_id, None)

    def _on_binary(self, data):
        with self._pending_lock:
            pending = self._last_binary_holder
            self._last_binary_holder = None
            if pending is None:
                return
            pending.binary = data
            pending.event.set()
            msg_id = pending.json.get("id") if pending.json else None
            if msg_id is not None:
                self._pending.pop(msg_id, None)

    # ---------------- send ----------------

    def _send(self, cmd, args=None, timeout=None):
        if not self._connected or self._ws is None:
            why = self._last_error
            if why:
                raise ConnectionError("not connected (%s). Call connect() again." % why)
            raise ConnectionError("not connected. Call connect() first.")

        msg_id = next(self._id_counter)
        pending = _Pending()
        with self._pending_lock:
            self._pending[msg_id] = pending

        envelope = {"cmd": cmd, "id": msg_id, "args": args or {}}
        try:
            with self._send_lock:
                self._ws.send(json.dumps(envelope))
        except Exception as ex:
            with self._pending_lock:
                self._pending.pop(msg_id, None)
            raise ConnectionError("send failed: %s" % ex)

        wait = timeout if timeout is not None else self._default_timeout
        if not pending.event.wait(wait):
            with self._pending_lock:
                self._pending.pop(msg_id, None)
            raise TimeoutError("timeout waiting for reply to %s id=%d" % (cmd, msg_id))

        if pending.error is not None:
            raise pending.error

        reply = pending.json or {}
        err = reply.get("error")
        if err:
            raise VisionServerError(err.get("code", "INTERNAL"), err.get("msg", ""))

        if pending.needs_binary:
            return reply, pending.binary
        return reply

    # ---------------- commands ----------------

    def hello(self, timeout=None):
        return self._send("hello", {}, timeout=timeout)

    # ---------------- binary follow-frame send -------------------------
    #
    # Local files (images, ML weights) ship inline with the call that
    # consumes them — there's no separate upload step and no server
    # filesystem staging. Bytes live only as long as the Detection that
    # uses them.

    def _send_with_binary(self, cmd, args, data, timeout=None):
        """Send a JSON envelope flagged binary_follows=true plus a
        binary frame. Returns the JSON reply (no binary expected back)."""
        if not self._connected or self._ws is None:
            raise ConnectionError("not connected. Call connect() first.")
        if not data:
            raise ValueError("binary frame must be non-empty")
        msg_id = next(self._id_counter)
        pending = _Pending()
        with self._pending_lock:
            self._pending[msg_id] = pending
        envelope = {"cmd": cmd, "id": msg_id, "args": args or {}, "binary_follows": True}
        try:
            with self._send_lock:
                self._ws.send(json.dumps(envelope))
                self._ws.send_binary(data)
        except Exception as ex:
            with self._pending_lock:
                self._pending.pop(msg_id, None)
            raise ConnectionError("binary send failed: %s" % ex)

        wait = timeout if timeout is not None else max(60, self._default_timeout)
        if not pending.event.wait(wait):
            with self._pending_lock:
                self._pending.pop(msg_id, None)
            raise TimeoutError("timeout waiting for reply to %s id=%d" % (cmd, msg_id))
        if pending.error is not None:
            raise pending.error
        reply = pending.json or {}
        err = reply.get("error")
        if err:
            raise VisionServerError(err.get("code", "INTERNAL"), err.get("msg", ""))
        return reply

    def camera_list(self, timeout=None):
        """Hardware discovery — RealSense devices currently attached to USB."""
        return self._send("camera_list", {}, timeout=timeout).get("devices", [])

    def camera_add(self, serial_number, timeout=None, **connect_kwargs):
        args = dict(connect_kwargs)
        args["serial_number"] = serial_number
        return self._send("camera_add", args, timeout=timeout)

    def camera_remove(self, serial_number, timeout=None):
        return self._send("camera_remove", {"serial_number": serial_number}, timeout=timeout)

    def robot_add(self, host, port=443, timeout_connect=5, model="dorna_ta", config=None, timeout=None):
        args = {"host": host, "port": port, "timeout": timeout_connect, "model": model}
        if config is not None:
            args["config"] = config
        return self._send("robot_add", args, timeout=timeout)

    def robot_remove(self, host, timeout=None):
        return self._send("robot_remove", {"host": host}, timeout=timeout)

    def detection_add(self, name, camera_serial_number=None, robot_host=None, timeout=None, **detection_kwargs):
        """
        Create a server-side Detection.

        If `detection={"cmd":"od|cls|kp", "path":"<local file>"}` and the
        path resolves to an actual file on this machine, the file's
        bytes are shipped inline with the call — the server loads them
        into the Detection then drops them. The path string in the
        envelope becomes a filename hint so the loader picks the right
        suffix; nothing is staged on the server's filesystem.
        """
        args = dict(detection_kwargs)
        args["name"] = name
        if camera_serial_number is not None:
            args["camera_serial_number"] = camera_serial_number
        if robot_host is not None:
            args["robot_host"] = robot_host

        det_pkg = args.get("detection") or {}
        path = det_pkg.get("path")
        if path and isinstance(path, str) and os.path.isfile(path):
            with open(path, "rb") as f:
                model_bytes = f.read()
            # Replace the absolute local path with just the basename —
            # it's only a filename hint to the server now.
            new_det = dict(det_pkg)
            new_det["path"] = os.path.basename(path)
            args["detection"] = new_det
            return self._send_with_binary("detection_add", args, model_bytes, timeout=timeout)
        return self._send("detection_add", args, timeout=timeout)

    def detection_run(self, name, use_last=False, timeout=None, **run_kwargs):
        args = dict(run_kwargs)
        args["name"] = name
        if use_last:
            args["use_last"] = True
        reply = self._send("detection_run", args, timeout=timeout)
        return reply.get("valid", [])

    def camera_get_img(self, serial_number, type="color_img", quality=75, timeout=None):
        reply, binary = self._send(
            "camera_get_img",
            {"serial_number": serial_number, "type": type, "quality": quality},
            timeout=timeout,
        )
        return binary, {k: v for k, v in reply.items() if k not in ("id", "stat", "binary_follows")}

    def detection_get_img(self, name, type="img", quality=85, timeout=None):
        reply, binary = self._send(
            "detection_get_img",
            {"name": name, "type": type, "quality": quality},
            timeout=timeout,
        )
        return binary, {k: v for k, v in reply.items() if k not in ("id", "stat", "binary_follows")}

    def detection_xyz(self, name, pxl, timeout=None):
        return self._send("detection_xyz", {"name": name, "pxl": list(pxl)}, timeout=timeout).get("xyz")

    def detection_pixel(self, name, xyz, timeout=None):
        return self._send("detection_pixel", {"name": name, "xyz": list(xyz)}, timeout=timeout).get("pxl")

    def detection_grasp(self, name, target_id, target_rvec, gripper_opening,
                        finger_wdith, finger_location,
                        mask_type="bb", prune_factor=2,
                        num_steps=360, search_angle=(0, 360),
                        timeout=None):
        args = {
            "name": name,
            "target_id": target_id,
            "target_rvec": list(target_rvec),
            "gripper_opening": gripper_opening,
            "finger_wdith": finger_wdith,
            "finger_location": list(finger_location),
            "mask_type": mask_type,
            "prune_factor": prune_factor,
            "num_steps": num_steps,
            "search_angle": list(search_angle),
        }
        return self._send("detection_grasp", args, timeout=timeout).get("rvec")

    def detection_remove(self, name, timeout=None):
        return self._send("detection_remove", {"name": name}, timeout=timeout)

    def detection_list(self, timeout=None):
        """Detections in this client's session."""
        return self._send("detection_list", {}, timeout=timeout).get("detections", [])

    # ---------------- dynamic proxies ----------------

    def detection(self, name):
        """Proxy to a server-side Detection; any method call forwards via RPC."""
        return _ObjectProxy(self, "detection", name)

    def camera(self, serial_number):
        """Proxy to a server-side Camera; any method call forwards via RPC."""
        return _ObjectProxy(self, "camera", serial_number)

    def robot(self, host):
        """Proxy to a server-side Dorna; any method call forwards via RPC."""
        return _ObjectProxy(self, "robot", host)

    def _call(self, target, name, method, args=None, kwargs=None, timeout=None):
        kw = dict(kwargs or {})
        # Auto-ship local files: if `data` is a string pointing to an
        # existing file (typical: detection.run(data="img/foo.jpg")),
        # read the bytes and send them as a binary follow-frame. The
        # server's `call` handler decodes them straight into the
        # detection's frame buffer — no temp file, no disk staging.
        data_val = kw.get("data")
        if isinstance(data_val, (bytes, bytearray)):
            payload = bytes(data_val)
            kw["data"] = "<binary>"      # placeholder; server overwrites with bytes
            return self._send_with_binary("call", {
                "target": target,
                "name": name,
                "method": method,
                "args": list(args or []),
                "kwargs": kw,
            }, payload, timeout=timeout).get("result")
        if isinstance(data_val, str) and os.path.isfile(data_val):
            with open(data_val, "rb") as f:
                payload = f.read()
            kw["data"] = os.path.basename(data_val)   # just a hint
            return self._send_with_binary("call", {
                "target": target,
                "name": name,
                "method": method,
                "args": list(args or []),
                "kwargs": kw,
            }, payload, timeout=timeout).get("result")
        reply = self._send("call", {
            "target": target,
            "name": name,
            "method": method,
            "args": list(args or []),
            "kwargs": kw,
        }, timeout=timeout)
        return reply.get("result")

    # ---------------- context manager ----------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class _ObjectProxy(object):
    """
    Proxy to a pooled server-side object (Detection, Camera, or Dorna).

    Any attribute access returns a callable that forwards to the server via
    the `call` RPC. So any method on the underlying class (existing or newly
    added later) is automatically callable without client changes.

    Plain attributes work too — `det.retval()` (with parens) returns the
    attribute value. Numpy arrays larger than a small threshold are replaced
    with a short {"_placeholder":"ndarray","shape":...,"dtype":...} stub so
    retval and similar structures stay compact. Use `.get_image(...)` to
    fetch actual image bytes.

    Usage:
        vc.detection("d1").run()
        vc.detection("d1").xyz([100, 200])
        vc.detection("d1").retval()
        vc.detection("d1").get_img("img")      # binary JPEG (not via proxy RPC)
        vc.camera(sn).set_exposure(1000)
        vc.robot("r1").joint()

    Pass `_timeout=<seconds>` as a keyword argument to override the call's
    wait time (the underscore avoids colliding with any real method kwarg).
    """

    __slots__ = ("_client", "_target", "_name")

    def __init__(self, client, target, name):
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_name", name)

    def get_img(self, type="img", quality=85, _timeout=None):
        """
        Detection-only: fetch image bytes over the binary channel. Returns
        (jpeg_bytes, meta_dict). Other targets (camera/robot) don't serve
        images — on them this raises.

        type: "img" | "img_roi" | "img_thr" | "color_img" | "depth_img" | "ir_img"
        """
        if self._target != "detection":
            raise AttributeError("get_img is only valid on a detection proxy")
        return self._client.detection_get_img(self._name, type=type, quality=quality, timeout=_timeout)

    def __getattr__(self, method):
        if method.startswith("_"):
            raise AttributeError(method)
        client = self._client
        target = self._target
        name = self._name

        def invoke(*args, **kwargs):
            timeout = kwargs.pop("_timeout", None)
            return client._call(target, name, method, list(args), kwargs, timeout=timeout)

        invoke.__name__ = method
        return invoke

    def __repr__(self):
        return "<%s(%r)>" % (self._target, self._name)


__all__ = ["VisionClient", "VisionServerError", "DEFAULT_TIMEOUT", "__version__"]
