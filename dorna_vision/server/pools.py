import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from camera import Camera, UEyeXS
from dorna2 import Dorna

from dorna_devices import MQTTDeviceAdapter, AutoRecover


log = logging.getLogger(__name__)

# Camera type registry — ``camera_add`` picks the driver via the "type"
# key in camera_cfg (default "d405", the RealSense). Every driver
# duck-types the same Device-protocol + capture surface, so the pool,
# the MQTT adapter and Detection treat them identically.
CAMERA_TYPES = {
    "d405": Camera,
    "ueye_xs": UEyeXS,
}


class _AutoRecoveringCamera:
    """Thin wrapper that routes recover() through an AutoRecover loop.

    MQTTDeviceAdapter calls ``device.recover()`` on operator-initiated
    cmds and expects a bool back. We swap the single-attempt behavior
    for a non-blocking trigger of the retry loop — same path as hotplug
    recovery, so operator and automatic flows share one mechanism.
    Returns True (queued); the actual outcome flows through state events.
    """

    def __init__(self, cam, auto_recover):
        self._cam = cam
        self._auto = auto_recover

    def __getattr__(self, name):
        # Delegate everything not overridden to the real Camera.
        return getattr(self._cam, name)

    def recover(self):
        self._auto.trigger()
        return True


class CameraPool(object):
    """
    Ref-counted pool of camera.Camera instances keyed by serial number.

    - acquire() connects on first add, increments ref on subsequent adds
    - release() decrements ref and closes when zero
    - executor(serial_number) returns a single-worker executor scoped to that
      camera so frame grabs and detections on the same camera serialize,
      while different cameras run in parallel

    Each acquired camera is also wrapped in an MQTTDeviceAdapter that
    publishes its health (info + state) per docs/device-guide.md, and
    an AutoRecover loop that auto-retries reconnection when the camera's
    USB comes back. Pass ``mqtt_enabled=False`` to disable (e.g. unit
    tests, dev boxes with no broker reachable).
    """

    def __init__(
        self,
        *,
        mqtt_enabled: bool = True,
        mqtt_broker_host: Optional[str] = None,
        mqtt_broker_port: Optional[int] = None,
        mqtt_critical: bool = True,
    ):
        self._lock = threading.Lock()
        self._cameras = {}      # serial_number -> Camera | UEyeXS
        self._refs = {}         # serial_number -> int
        self._executors = {}    # serial_number -> ThreadPoolExecutor(max_workers=1)
        self._adapters = {}     # serial_number -> MQTTDeviceAdapter
        self._recovers = {}     # serial_number -> AutoRecover
        self._types = {}        # serial_number -> "d405" | "ueye_xs"

        self._mqtt_enabled = mqtt_enabled
        self._mqtt_broker_host = mqtt_broker_host
        self._mqtt_broker_port = mqtt_broker_port
        self._mqtt_critical = mqtt_critical

        # Touch the Camera class once so its singleton rs.context() and
        # hotplug callback are wired up at server start. After this, the
        # cached rs._all_device list updates automatically when devices
        # are plugged in / unplugged.
        Camera()

    def list_devices(self):
        # RealSense: no refresh call needed — the hotplug callback registered
        # inside Camera() keeps rs._all_device current. uEye: enumerated on
        # demand (returns [] when the IDS SDK isn't installed).
        out = [{k: v for k, v in d.items() if k != "obj"} | {"camera_type": "d405"}
               for d in Camera().all_device()]
        out.extend(UEyeXS.all_device())
        return out

    def camera_type(self, serial_number):
        with self._lock:
            return self._types.get(serial_number)

    def acquire(self, serial_number, **connect_kwargs):
        ctype = connect_kwargs.pop("type", "d405") or "d405"
        cls = CAMERA_TYPES.get(ctype)
        if cls is None:
            raise ValueError(
                "unknown camera type %r (known: %s)" % (ctype, sorted(CAMERA_TYPES)))
        with self._lock:
            if serial_number in self._cameras:
                self._refs[serial_number] += 1
                return self._cameras[serial_number]

            cam = cls()
            ok = cam.connect(serial_number=serial_number, **connect_kwargs)
            if not ok:
                try:
                    cam.close()
                except Exception:
                    pass
                raise RuntimeError("camera connect failed for serial_number=%s" % serial_number)

            self._cameras[serial_number] = cam
            self._refs[serial_number] = 1
            self._types[serial_number] = ctype
            self._executors[serial_number] = ThreadPoolExecutor(max_workers=1)

            # Self-healing reconnect loop. The camera SDK fires
            # ``on_hardware_available`` from its hotplug callback when our
            # USB serial reappears on the bus; AutoRecover.trigger() runs
            # cam.recover() with exponential backoff and surfaces attempt
            # counts via cam._set_state. Operator-initiated recoveries
            # (cmd handler) re-use the same loop — see _make_recoverable
            # below — so manual + automatic paths share one mechanism.
            recover = AutoRecover(
                recover_fn=cam.recover,
                set_status=cam._set_state,
                log_label=f"camera:{serial_number}",
            )
            cam.on_hardware_available(recover.trigger)
            self._recovers[serial_number] = recover

            if self._mqtt_enabled:
                try:
                    # Wrap the camera so MQTTDeviceAdapter's cmd handler
                    # routes recover() through the AutoRecover loop instead
                    # of running a single blocking attempt. Same operator
                    # UX as a hotplug-triggered recover.
                    publishable = _AutoRecoveringCamera(cam, recover)
                    adapter = MQTTDeviceAdapter(
                        publishable,
                        kind="camera",
                        critical=self._mqtt_critical,
                        meta={"serial_number": serial_number},
                        broker_host=self._mqtt_broker_host,
                        broker_port=self._mqtt_broker_port,
                    )
                    self._adapters[serial_number] = adapter
                except Exception:
                    # Adapter failure must NOT take down the pool — the
                    # camera itself is still usable for detection work.
                    log.exception(
                        "CameraPool: failed to attach MQTT adapter for %s; "
                        "camera will run without health publishing.",
                        serial_number,
                    )

            return cam

    def set_broker(self, host, port=1883):
        """Point device-state publishing at a (new) broker — the
        zero-config site-bus handshake: the workspace tells this unit
        "publish to my bus". Rebuilds the adapter of every held
        camera; future acquires use the new broker automatically."""
        with self._lock:
            if (host, int(port)) == (self._mqtt_broker_host, self._mqtt_broker_port):
                return
            self._mqtt_broker_host, self._mqtt_broker_port = host, int(port)
            if not self._mqtt_enabled:
                return
            for sn, cam in self._cameras.items():
                old_adapter = self._adapters.pop(sn, None)
                if old_adapter is not None:
                    try:
                        old_adapter.close()
                    except Exception:
                        pass
                try:
                    publishable = _AutoRecoveringCamera(cam, self._recovers.get(sn))
                    self._adapters[sn] = MQTTDeviceAdapter(
                        publishable,
                        kind="camera",
                        critical=self._mqtt_critical,
                        meta={"serial_number": sn},
                        broker_host=host,
                        broker_port=int(port),
                    )
                except Exception:
                    log.exception(
                        "CameraPool: failed to re-point MQTT adapter for %s", sn)
            log.info("CameraPool: device bus -> %s:%s", host, port)

    def release(self, serial_number):
        with self._lock:
            if serial_number not in self._cameras:
                return
            self._refs[serial_number] -= 1
            if self._refs[serial_number] > 0:
                return

            cam = self._cameras.pop(serial_number)
            self._refs.pop(serial_number, None)
            self._types.pop(serial_number, None)
            executor = self._executors.pop(serial_number, None)
            adapter = self._adapters.pop(serial_number, None)
            recover = self._recovers.pop(serial_number, None)

        if recover is not None:
            try:
                recover.stop()
            except Exception:
                log.exception("CameraPool: recover.stop() failed for %s",
                              serial_number)
        if adapter is not None:
            try:
                adapter.close()
            except Exception:
                log.exception("CameraPool: adapter.close() failed for %s",
                              serial_number)
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
            adapters = dict(self._adapters)
            recovers = dict(self._recovers)
            self._cameras.clear()
            self._refs.clear()
            self._executors.clear()
            self._adapters.clear()
            self._recovers.clear()

        for recover in recovers.values():
            try:
                recover.stop()
            except Exception:
                pass
        for adapter in adapters.values():
            try:
                adapter.close()
            except Exception:
                pass
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
