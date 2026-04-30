import threading

from dorna_vision.detect import Detection


class ClientSession(object):
    """
    Per-websocket-connection state.

    Owns:
      - a detection pool keyed by client-supplied name, scoped to this client

    Cameras and robots are server-global. Adding them does not tie them to
    this client, and disconnecting does not release them. They live until
    somebody explicitly calls camera_remove / robot_remove, or the server
    shuts down.
    """

    def __init__(self, camera_pool, robot_pool):
        self.camera_pool = camera_pool
        self.robot_pool = robot_pool

        self._lock = threading.Lock()
        self._detections = {}                      # name -> Detection
        self._detection_camera_serial = {}         # name -> camera serial_number (for executor lookup)

    # ---------------- camera (server-global) ----------------

    def camera_add(self, serial_number, **connect_kwargs):
        return self.camera_pool.acquire(serial_number, **connect_kwargs)

    def camera_remove(self, serial_number):
        self.camera_pool.release(serial_number)
        return True

    # ---------------- robot (server-global) ----------------

    def robot_add(self, host, port=443, timeout=5, model="dorna_ta", config=None):
        return self.robot_pool.acquire(host=host, port=port, timeout=timeout, model=model, config=config)

    def robot_remove(self, host):
        self.robot_pool.release(host)
        return True

    # ---------------- detection (per-client) ----------------

    def detection_add(self, name, camera_serial_number=None, robot_host=None, **detection_kwargs):
        with self._lock:
            if name in self._detections:
                raise ValueError("detection name already exists: %s" % name)

        cam = None
        if camera_serial_number is not None:
            cam = self.camera_pool.get(camera_serial_number)
            if cam is None:
                raise ValueError("camera not found: %s" % camera_serial_number)

        robot = None
        if robot_host is not None:
            robot = self.robot_pool.get(robot_host)
            if robot is None:
                # Lazy acquire with defaults (port=443, model=dorna_ta).
                # For non-defaults, the client can call robot_add explicitly first.
                robot = self.robot_pool.acquire(host=robot_host)

        det = Detection(camera=cam, robot=robot, **detection_kwargs)

        with self._lock:
            self._detections[name] = det
            self._detection_camera_serial[name] = camera_serial_number
        return det

    def detection_get(self, name):
        with self._lock:
            det = self._detections.get(name)
        if det is None:
            raise ValueError("detection not found: %s" % name)
        return det

    def detection_camera_serial_number(self, name):
        with self._lock:
            return self._detection_camera_serial.get(name)

    def detection_remove(self, name):
        with self._lock:
            det = self._detections.pop(name, None)
            self._detection_camera_serial.pop(name, None)
        if det is None:
            return False
        try:
            det.close()
        except Exception:
            pass
        return True

    def detection_list(self):
        with self._lock:
            out = []
            for name, det in self._detections.items():
                cmd = None
                try:
                    if isinstance(det.detection, dict):
                        cmd = det.detection.get("cmd")
                except Exception:
                    pass
                out.append({
                    "name": name,
                    "camera_serial_number": self._detection_camera_serial.get(name),
                    "cmd": cmd,
                })
            return out

    # ---------------- cleanup ----------------

    def close(self):
        """Called when the websocket disconnects. Releases detections only."""
        with self._lock:
            dets = dict(self._detections)
            self._detections.clear()
            self._detection_camera_serial.clear()

        for det in dets.values():
            try:
                det.close()
            except Exception:
                pass
