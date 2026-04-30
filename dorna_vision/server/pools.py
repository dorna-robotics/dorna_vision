import threading
from concurrent.futures import ThreadPoolExecutor

from camera import Camera
from dorna2 import Dorna


class CameraPool(object):
    """
    Ref-counted pool of camera.Camera instances keyed by serial number.

    - acquire() connects on first add, increments ref on subsequent adds
    - release() decrements ref and closes when zero
    - executor(serial_number) returns a single-worker executor scoped to that
      camera so frame grabs and detections on the same camera serialize,
      while different cameras run in parallel
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cameras = {}      # serial_number -> Camera
        self._refs = {}         # serial_number -> int
        self._executors = {}    # serial_number -> ThreadPoolExecutor(max_workers=1)

        # Touch the Camera class once so its singleton rs.context() and
        # hotplug callback are wired up at server start. After this, the
        # cached rs._all_device list updates automatically when devices
        # are plugged in / unplugged.
        Camera()

    def list_devices(self):
        # No refresh call needed — the hotplug callback registered inside
        # Camera() keeps rs._all_device current. Just read it.
        return [{k: v for k, v in d.items() if k != "obj"} for d in Camera().all_device()]

    def acquire(self, serial_number, **connect_kwargs):
        with self._lock:
            if serial_number in self._cameras:
                self._refs[serial_number] += 1
                return self._cameras[serial_number]

            cam = Camera()
            ok = cam.connect(serial_number=serial_number, **connect_kwargs)
            if not ok:
                try:
                    cam.close()
                except Exception:
                    pass
                raise RuntimeError("camera connect failed for serial_number=%s" % serial_number)

            self._cameras[serial_number] = cam
            self._refs[serial_number] = 1
            self._executors[serial_number] = ThreadPoolExecutor(max_workers=1)
            return cam

    def release(self, serial_number):
        with self._lock:
            if serial_number not in self._cameras:
                return
            self._refs[serial_number] -= 1
            if self._refs[serial_number] > 0:
                return

            cam = self._cameras.pop(serial_number)
            self._refs.pop(serial_number, None)
            executor = self._executors.pop(serial_number, None)

        if executor is not None:
            executor.shutdown(wait=True)
        try:
            cam.close()
        except Exception:
            pass

    def get(self, serial_number):
        with self._lock:
            return self._cameras.get(serial_number)

    def executor(self, serial_number):
        with self._lock:
            return self._executors.get(serial_number)

    def list_pool_keys(self):
        with self._lock:
            return list(self._cameras.keys())

    def shutdown(self):
        with self._lock:
            serials = list(self._cameras.keys())
            cams = dict(self._cameras)
            execs = dict(self._executors)
            self._cameras.clear()
            self._refs.clear()
            self._executors.clear()

        for e in execs.values():
            e.shutdown(wait=True)
        for serial_number in serials:
            try:
                cams[serial_number].close()
            except Exception:
                pass


class RobotPool(object):
    """
    Ref-counted pool of dorna2.Dorna instances keyed by host (IP).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._robots = {}   # host -> Dorna
        self._refs = {}     # host -> int
        self._info = {}     # host -> {"port": int, "model": str}

    def acquire(self, host, port=443, timeout=5, model="dorna_ta", config=None):
        with self._lock:
            if host in self._robots:
                self._refs[host] += 1
                return self._robots[host]

            robot = Dorna(config=config, model=model)
            ok = robot.connect(host=host, port=port, timeout=timeout)
            if not ok:
                try:
                    robot.close()
                except Exception:
                    pass
                raise RuntimeError("robot connect failed for host=%s" % host)

            self._robots[host] = robot
            self._refs[host] = 1
            self._info[host] = {"port": port, "model": model}
            return robot

    def release(self, host):
        with self._lock:
            if host not in self._robots:
                return
            self._refs[host] -= 1
            if self._refs[host] > 0:
                return
            robot = self._robots.pop(host)
            self._refs.pop(host, None)
            self._info.pop(host, None)

        try:
            robot.close()
        except Exception:
            pass

    def get(self, host):
        with self._lock:
            return self._robots.get(host)

    def shutdown(self):
        with self._lock:
            robots = dict(self._robots)
            self._robots.clear()
            self._refs.clear()
        for r in robots.values():
            try:
                r.close()
            except Exception:
                pass
