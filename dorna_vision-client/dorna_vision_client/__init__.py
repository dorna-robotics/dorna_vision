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
import logging
from concurrent.futures import ThreadPoolExecutor
import os
import threading
import time

import websocket  # websocket-client


__version__ = "0.1.0"

DEFAULT_TIMEOUT = 10.0


def _hand_back(path):
    """Under sudo, give a file or folder this process created back to the
    invoking user — same rule as the server's save_img. Never raises."""
    try:
        uid = int(os.environ.get("SUDO_UID", -1))
        gid = int(os.environ.get("SUDO_GID", -1))
        if uid >= 0:
            os.chown(path, uid, gid)
    except Exception:
        pass


def _client_save(event, data):
    """Write a detection_img event's bytes where its ``target`` says — the
    rules of the detection's save_img, resolved on THIS computer:
      True              -> output/<timestamp>.jpg (roi_<timestamp>.jpg)
      "folder/" or dir  -> <timestamp>.jpg / roi_<timestamp>.jpg inside it
      "file.ext"        -> that file, overwritten every run
    Folders are created. Returns the path written."""
    target = event.get("target")
    prefix = "roi_" if event.get("type") == "img_roi" else ""
    stem = prefix + str(int(event.get("timestamp") or time.time()))
    ext = "." + (event.get("encoding") or "jpg")
    if isinstance(target, str):
        target = os.path.expanduser(target)
        if target.endswith(("/", os.sep)) or os.path.isdir(target):
            folder, path = target, os.path.join(target, stem + ext)
        else:
            folder, path = os.path.dirname(os.path.abspath(target)), target
    else:
        folder = "output"
        path = os.path.join(folder, stem + ext)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
        _hand_back(folder)
    with open(path, "wb") as fh:
        fh.write(data)
    _hand_back(path)
    return path


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
        # Unsolicited server events ({"event": ..., no "id"}). One that says
        # binary_follows waits here for its frame — a SEPARATE holder from
        # replies', so an event and a reply never take each other's binary.
        self._event_binary_holder = None
        self._event_listeners = []
        # Events are delivered — files written, listeners called — on ONE
        # background thread, in arrival order, so the reader never blocks
        # on a disk write and replies are never delayed by an event.
        self._event_executor = None
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
            self._event_binary_holder = None

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

    def on_event(self, fn):
        """Register ``fn(event, binary)`` for unsolicited server events:
        ``event`` is the envelope dict (``event["event"]`` names it, e.g.
        "detection_img"), ``binary`` the frame that came with it, or None.

        A detection_img event's file (display.client_save_img /
        client_save_img_roi) is already written when ``fn`` runs, and
        ``event["path"]`` says where (None if the write failed). Listeners
        run on the client's event thread, in arrival order; an exception in
        ``fn`` is logged and never stops delivery. Returns ``fn``, so it
        also works as a decorator."""
        with self._pending_lock:
            self._event_listeners.append(fn)
        return fn

    def off_event(self, fn):
        """Remove a listener added with ``on_event``."""
        with self._pending_lock:
            if fn in self._event_listeners:
                self._event_listeners.remove(fn)

    def _emit_event(self, event, binary):
        with self._pending_lock:
            if self._event_executor is None:
                self._event_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="vision-events")
            ex = self._event_executor
        try:
            ex.submit(self._deliver_event, event, binary)
        except RuntimeError:            # client closing
            pass

    def _deliver_event(self, event, binary):
        if event.get("event") == "detection_img" and event.get("target") and binary is not None:
            try:
                event["path"] = _client_save(event, binary)
            except Exception:
                event["path"] = None
                logging.getLogger(__name__).exception(
                    "could not save %s %s to %r", event.get("name"),
                    event.get("type"), event.get("target"))
        with self._pending_lock:
            listeners = list(self._event_listeners)
        for fn in listeners:
            try:
                fn(event, binary)
            except Exception:
                logging.getLogger(__name__).exception(
                    "event listener failed on %s", event.get("event"))

    def _on_text(self, text):
        try:
            payload = json.loads(text)
        except Exception:
            return

        if "id" not in payload and "event" in payload:
            if payload.get("binary_follows"):
                with self._pending_lock:
                    self._event_binary_holder = payload
                return
            self._emit_event(payload, None)
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
        # The server writes every envelope + binary pair atomically, so a
        # binary belongs to whichever envelope announced one last.
        with self._pending_lock:
            event = self._event_binary_holder
            self._event_binary_holder = None
        if event is not None:
            self._emit_event(event, data)
            return
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
        """Hardware discovery — camera devices currently attached to USB
        (RealSense + uEye; each carries camera_type)."""
        return self._send("camera_list", {}, timeout=timeout).get("devices", [])

    def camera_add(self, serial_number, timeout=None, **connect_kwargs):
        args = dict(connect_kwargs)
        args["serial_number"] = serial_number
        return self._send("camera_add", args, timeout=timeout)

    def bus_connect(self, host=None, port=1883, timeout=None):
        """Point the unit's device-state publishing at a broker. With
        host=None the server uses THIS client's address — the calling
        workspace is the site's broker host (zero-config site bus)."""
        args = {"port": int(port)}
        if host:
            args["host"] = host
        return self._send("bus_connect", args, timeout=timeout)

    def camera_remove(self, serial_number, timeout=None):
        return self._send("camera_remove", {"serial_number": serial_number}, timeout=timeout)

    def camera_info(self, serial_number, timeout=None):
        """Live facts for a pooled camera: type, the stream that actually
        runs, the intrinsics in effect (labeled factory/override/nominal),
        and — for cameras with a focus surface — the focus state."""
        return self._send("camera_info", {"serial_number": serial_number}, timeout=timeout)

    def camera_recover(self, serial_number, timeout=60):
        """Trigger recovery on a pooled camera (the GUI's Recover button,
        over the API). Returns {ok, state, msg}."""
        return self._send("camera_recover", {"serial_number": serial_number}, timeout=timeout)

    def camera_focus(self, serial_number, mode=None, position=None, region=None,
                     method=None, timeout=None):
        """Focus control for cameras with a focus surface (uEye XS).

        Pass ONE of:
          mode="continuous"            SDK continuous autofocus
          mode="once"                  one-shot autofocus, then hold
          mode="manual", position=N    pin the lens position
          region=[x0, y0, x1, y1]      region focus: the camera's own AF
                                       pointed at the rect (~1-2 s), pinned
                                       where it lands. method="sweep" forces
                                       the deterministic manual-lens sweep
                                       (~15-30 s); "af" forces hardware AF;
                                       default "auto" tries af, falls back.

        Returns the reply dict; "focus" carries the camera's focus_info.
        """
        args = {"serial_number": serial_number}
        if region is not None:
            args["region"] = [int(v) for v in region]
            if method is not None:
                args["method"] = method
        else:
            if mode is not None:
                args["mode"] = mode
            if position is not None:
                args["position"] = int(position)
        # region sweeps far exceed the default reply timeout
        if timeout is None and region is not None:
            timeout = 120
        return self._send("camera_focus", args, timeout=timeout)

    def camera_exposure(self, serial_number, ms=None, auto=None, timeout=None):
        """Sensor exposure (integration time) — distinct from the detection
        pipeline's software `intensity`. ms=<value> pins it on cameras that
        allow manual exposure; auto=True re-enables auto; neither just
        reports. NOTE: the uEye XS ISP owns exposure (auto only) — passing
        ms there raises with that explanation. Returns
        {"exposure": <value>, "auto": bool}."""
        args = {"serial_number": serial_number}
        if ms is not None:
            args["ms"] = float(ms)
        elif auto:
            args["auto"] = True
        return self._send("camera_exposure", args, timeout=timeout)

    def camera_wb(self, serial_number, auto=None, hold=None, timeout=None):
        """White balance (uEye XS). auto=True — in-camera auto WB;
        hold=True — freeze WB at its current convergence (the bench
        recipe: let auto settle on the lit scene, then hold —
        deterministic color from then on). Neither just reports.
        The XS ISP rejects fixed kelvin/rgb — hold is the fixed-WB tool."""
        args = {"serial_number": serial_number}
        if auto:
            args["auto"] = True
        elif hold:
            args["hold"] = True
        return self._send("camera_wb", args, timeout=timeout)

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

    def detection_capture(self, name, data=None, camera_in_world=None, focus=None, frames_avg=None, timeout=None):
        """Capture a fresh atomic snapshot (camera frames + robot joint
        angles) for ``name`` and cache it on the server. Returns the
        full reply dict so callers can branch on ``ok`` without raising:

            {"name": ..., "ok": True,  "ts": <float>, "has_joint": bool}
            {"name": ..., "ok": False, "msg": "<error description>"}

        Pair with ``detection_run(name, use_last=True)`` so detection
        runs ONLY on a confirmed-fresh frame — no silent fallback to a
        stale cache. The orchestrator decides whether to retry / pause
        / surface to operator on ``ok=False``.

        ``data`` mirrors ``Detection.get_camera_data``:
          * ``None`` — live camera (default).
          * ``dict`` — pre-fetched payload (replay / cross-detection).
          * ``str``  — server-local image path. (File-shipping from the
            client uses the binary follow-frame path on ``call`` —
            distinct from this method.)

        ``frames_avg`` — aligned frame averaging at capture (noise
        ~÷√N): N>1 grabs N color frames, registers each to the first
        (sub-pixel translation — absorbs servo jitter) and averages;
        1 = single grab (default). Sticky per detection, like ``focus``.
        """
        args = {"name": name}
        if data is not None:
            args["data"] = data
        if camera_in_world is not None:
            args["camera_in_world"] = list(camera_in_world)
        if focus is not None:
            args["focus"] = focus
        if frames_avg is not None:
            args["frames_avg"] = int(frames_avg)
        return self._send("detection_capture", args, timeout=timeout)

    def camera_get_img(self, serial_number, type="color_img", quality=75, timeout=None):
        reply, binary = self._send(
            "camera_get_img",
            {"serial_number": serial_number, "type": type, "quality": quality},
            timeout=timeout,
        )
        return binary, {k: v for k, v in reply.items() if k not in ("id", "stat", "binary_follows")}

    def detection_get_img(self, name, type="img", quality=85, max_side=None, timeout=None):
        """The detection's last image as (jpeg_bytes, meta). ``max_side``
        asks the server to downscale so the longest side is at most that
        many px BEFORE encoding — a 3072x2048 annotated frame is ~750 KB
        at full size; a 1536 cap is a quarter of the bytes on the wire.
        Preview-only: coordinates from other calls stay full-resolution."""
        args = {"name": name, "type": type, "quality": quality}
        if max_side:
            args["max_side"] = int(max_side)
        reply, binary = self._send("detection_get_img", args, timeout=timeout)
        return binary, {k: v for k, v in reply.items() if k not in ("id", "stat", "binary_follows")}

    def detection_xyz(self, name, pxl, timeout=None):
        return self._send("detection_xyz", {"name": name, "pxl": list(pxl)}, timeout=timeout).get("xyz")

    def detection_pixel(self, name, xyz, timeout=None):
        return self._send("detection_pixel", {"name": name, "xyz": list(xyz)}, timeout=timeout).get("pxl")

    def detection_box_corners(self, name, box, K=None, D=None, timeout=None):
        # box = [x, y, z, a, b, c, w, d, h]: bottom-plane-center pose + extents.
        # Returns the convex-hull pixel polygon of the box's 8 projected corners
        # (the outer ROI silhouette). h>0 floor, h<0 ceiling.
        #
        # By default the intrinsics from the last run() are used. Pass your own
        # K (3x3) and D (distortion coeffs) to project against supplied
        # intrinsics instead.
        args = {"name": name, "box": list(box)}
        if K is not None and D is not None:
            args["K"] = [list(r) for r in K]
            args["D"] = list(D)
        return self._send("detection_box_corners", args, timeout=timeout).get("corners")

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
        vc.detection("d1").save_img("a.jpg")   # ...written on THIS computer
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

    def save_img(self, path, type="img", quality=100, _timeout=None):
        """
        Detection-only: fetch the image and write it on THIS computer — the
        one calling the API, not the vision unit (that is what the
        detection's own save_img / save_img_roi display options do).

            cnt.run()
            cnt.save_img("captures/a.jpg", type="img_roi")

        Full resolution, JPEG at ``quality`` (100 by default: saved images
        are usually a dataset). ``path`` naming follows the server-side
        save_img: a folder (trailing "/" or an existing directory) gets
        ``<timestamp>.jpg``, ``roi_<timestamp>.jpg`` for img_roi; anything
        else is the file, overwritten. The bytes are JPEG whatever the
        extension. Folders are created as needed. Returns the path written.

        type: "img" | "img_roi" | "img_thr" | "color_img" | "depth_img" | "ir_img"
        """
        jpeg, _meta = self.get_img(type=type, quality=quality, _timeout=_timeout)
        path = os.path.expanduser(str(path))
        if path.endswith(("/", os.sep)) or os.path.isdir(path):
            stem = str(int(time.time() * 1000))
            path = os.path.join(path, ("roi_" if type == "img_roi" else "") + stem + ".jpg")
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(jpeg)
        return path

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
