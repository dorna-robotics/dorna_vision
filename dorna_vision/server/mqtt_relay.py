"""MQTTDeviceObserver — vision-server-side subscriber that turns the MQTT
device bus into local web-client push events.

The vision server publishes its cameras to MQTT via :class:`MQTTDeviceAdapter`.
That same server's web UI also wants to display device health. Rather than
short-circuit through the local Camera object (which would mean the web UI
gets data from a different source than every other subscriber), we route
the web UI **through the bus too** — single source of truth, identical
data path as the workspace orchestrator and any future dashboard.

This class subscribes to ``device/+/state`` and ``device/+/info``, caches
the latest payload per device, and invokes a broadcast callback on every
update. The vision server's ``app.py`` wires that callback to
``VisionWSHandler.broadcast`` so connected browsers see device events as
they happen on the bus.

Distinct from ``workspace.devices.MQTTOrchestrator`` (which adds runtime
pause/resume policy and recover/release commands). This class is purely
observe + cache + broadcast — production behavior for the vision web UI.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable, Optional

import paho.mqtt.client as mqtt


log = logging.getLogger(__name__)


DEFAULT_BROKER_HOST = os.environ.get("DEVICE_MQTT_HOST", "localhost")
DEFAULT_BROKER_PORT = int(os.environ.get("DEVICE_MQTT_PORT", "1883"))


class MQTTDeviceObserver:
    """Subscribe to the device bus, cache latest state, fan out to a callback.

    Args:
        broadcast: Called for every incoming state OR info update with a
            JSON-serializable dict. Vision server's app.py wires this to
            ``VisionWSHandler.broadcast``.
        broker_host / broker_port: Broker location. Defaults to env
            ``DEVICE_MQTT_HOST`` / ``DEVICE_MQTT_PORT``, then localhost:1883.
        client_id: Optional MQTT client id; defaults to a unique
            ``vision-observer-<uuid>``.
        client_factory: Internal hook for tests — returns an object with
            paho's client interface.
    """

    def __init__(
        self,
        broadcast: Callable[[dict[str, Any]], None],
        *,
        broker_host: Optional[str] = None,
        broker_port: Optional[int] = None,
        client_id: Optional[str] = None,
        client_factory: Optional[Callable[[str], Any]] = None,
    ):
        import uuid as _uuid

        self._broadcast = broadcast
        self.broker_host = broker_host if broker_host is not None else DEFAULT_BROKER_HOST
        self.broker_port = broker_port if broker_port is not None else DEFAULT_BROKER_PORT

        self._lock = threading.RLock()
        self._cache: dict[str, dict[str, Any]] = {}  # device_id -> latest snapshot
        self._closed = False

        cid = client_id or f"vision-observer-{_uuid.uuid4().hex[:6]}"
        if client_factory is None:
            client_factory = lambda c: mqtt.Client(  # noqa: E731
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=c,
            )
        self.client = client_factory(cid)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        try:
            self.client.connect(self.broker_host, self.broker_port, keepalive=30)
        except Exception as ex:
            log.warning(
                "MQTTDeviceObserver: initial connect to %s:%s failed (%s); "
                "will retry in background.",
                self.broker_host, self.broker_port, ex,
            )
        try:
            self.client.loop_start()
        except Exception:
            log.exception("MQTTDeviceObserver: loop_start failed")

    # ── paho callbacks (v2 API) ───────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        try:
            client.subscribe([
                ("device/+/info", 1),
                ("device/+/state", 1),
            ])
            log.info("MQTTDeviceObserver: connected to %s:%s, subscribed.",
                     self.broker_host, self.broker_port)
        except Exception:
            log.exception("MQTTDeviceObserver: subscribe failed on connect")

    def _on_disconnect(self, client, userdata, *args, **kwargs):
        log.info("MQTTDeviceObserver: disconnected (paho will retry)")

    def _on_message(self, client, userdata, message):
        try:
            self._dispatch(message.topic, message.payload)
        except Exception:
            log.exception("MQTTDeviceObserver: failed to handle %s",
                          getattr(message, "topic", "<unknown>"))

    # ── Topic dispatch ────────────────────────────────────────────────────

    def _dispatch(self, topic: str, raw_payload: bytes) -> None:
        parts = topic.split("/")
        # ``device/<id>/info`` or ``device/<id>/state`` — same shape, two leaves.
        if len(parts) != 3 or parts[0] != "device":
            return
        leaf = parts[2]
        if leaf not in ("info", "state"):
            return
        device_id = parts[1]
        try:
            payload = json.loads(raw_payload.decode())
        except Exception:
            log.warning("MQTTDeviceObserver: bad JSON on %s", topic)
            return

        with self._lock:
            entry = self._cache.setdefault(device_id, {
                "id": device_id,
                "state": "down",
                "msg": "no state yet",
                "kind": "device",
                "critical": True,
                "meta": {},
                "ts": 0.0,
            })
            if leaf == "info":
                entry["kind"] = str(payload.get("kind", entry["kind"]))
                entry["critical"] = bool(payload.get("critical", entry["critical"]))
                entry["meta"] = dict(payload.get("meta", entry["meta"]))
            else:  # state
                entry["state"] = str(payload.get("state", entry["state"]))
                entry["msg"]   = str(payload.get("msg",   ""))
                entry["ts"]    = float(payload.get("ts",  time.time()))
            snapshot = dict(entry)

        # Build the broadcast event. Include `serial_number` derived from the
        # `<kind>:<natural-id>` convention so JS clients that already key by
        # serial keep working without a second lookup.
        serial = self._extract_natural_id(device_id)
        evt = {
            "type": "device_state" if leaf == "state" else "device_info",
            "id": device_id,
            "serial_number": serial,
            "state": snapshot["state"],
            "msg": snapshot["msg"],
            "kind": snapshot["kind"],
            "critical": snapshot["critical"],
            "meta": snapshot["meta"],
            "ts": snapshot["ts"],
        }
        try:
            self._broadcast(evt)
        except Exception:
            log.exception("MQTTDeviceObserver: broadcast callback raised")

    @staticmethod
    def _extract_natural_id(device_id: str) -> str:
        """``"camera:130322274110"`` → ``"130322274110"``. No prefix → id as-is."""
        if ":" in device_id:
            return device_id.split(":", 1)[1]
        return device_id

    # ── Public API ────────────────────────────────────────────────────────

    def snapshot(self) -> list[dict[str, Any]]:
        """Return a list of every device the observer has seen, latest state."""
        with self._lock:
            out = []
            for device_id, entry in self._cache.items():
                serial = self._extract_natural_id(device_id)
                out.append({
                    "type": "device_state",
                    "id": device_id,
                    "serial_number": serial,
                    **{k: entry[k] for k in ("state", "msg", "kind", "critical", "meta", "ts")},
                })
            return out

    def close(self) -> None:
        """Idempotent shutdown."""
        if self._closed:
            return
        self._closed = True
        try:
            self.client.loop_stop()
        except Exception:
            pass
        try:
            self.client.disconnect()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
