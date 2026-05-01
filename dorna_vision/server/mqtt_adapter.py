"""MQTTDeviceAdapter — publishes a Device-shaped object to MQTT per the spec.

See ``docs/device-mqtt-spec.md`` (in the workspace repo) for the wire
contract. The adapter wraps any object that exposes the structural
``Device`` shape — ``id``, ``state``, ``msg``, ``on_state_change(cb)``,
``recover() -> bool``, ``release()`` — and:

  * publishes ``device/<id>/info`` (retained) on connect,
  * publishes ``device/<id>/state`` (retained) on every state change,
  * configures a Last Will & Testament so the broker auto-marks the
    device down if this process disappears,
  * subscribes to ``device/<id>/cmd/{recover,release}`` and replies on
    ``.../reply`` with the round-trip result.

Adapter does not import workspace or any other orchestrator code —
it's a pure publisher. Anything subscribing to the spec's topics will
see this device.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Optional

import paho.mqtt.client as mqtt


log = logging.getLogger(__name__)


DEFAULT_BROKER_HOST = os.environ.get("DEVICE_MQTT_HOST", "localhost")
DEFAULT_BROKER_PORT = int(os.environ.get("DEVICE_MQTT_PORT", "1883"))


class MQTTDeviceAdapter:
    """Wrap a Device-shaped object and publish/subscribe per the MQTT spec.

    Args:
        device: Any object with attributes ``id``, ``state``, ``msg`` and
            methods ``on_state_change(cb)``, ``recover() -> bool``, ``release()``.
        kind: Short device family name, e.g. ``"camera"``, ``"printer"``.
            Published in the ``info`` payload so subscribers know what it is.
        critical: Whether this device's failure should pause the orchestrator.
            Published in ``info``; the orchestrator reads it to decide policy.
        meta: Free-form dict published in ``info`` (e.g. model, USB port).
        broker_host: MQTT broker host. Defaults to env ``DEVICE_MQTT_HOST``
            then ``"localhost"``.
        broker_port: MQTT broker port. Defaults to env ``DEVICE_MQTT_PORT``
            then ``1883``.
        client_id: Optional MQTT client id; defaults to ``"dev-<id>-<uuid>"``.
    """

    def __init__(
        self,
        device: Any,
        *,
        kind: str,
        critical: bool = True,
        meta: Optional[dict[str, Any]] = None,
        broker_host: Optional[str] = None,
        broker_port: Optional[int] = None,
        client_id: Optional[str] = None,
    ):
        self.device = device
        self.kind = kind
        self.critical = critical
        self.meta = dict(meta or {})
        self.broker_host = broker_host if broker_host is not None else DEFAULT_BROKER_HOST
        self.broker_port = broker_port if broker_port is not None else DEFAULT_BROKER_PORT

        device_id = getattr(device, "id", None)
        if not device_id:
            raise ValueError("device.id must be a non-empty string before adapter construction")
        device_id = str(device_id)
        # Normalize to the spec convention `<kind>:<natural-id>`. If the
        # device class already produces a prefixed id (recommended), this
        # is a no-op; if not, we prepend the kind so topic dumps stay
        # readable across multiple device families on the same bus.
        if ":" not in device_id:
            device_id = f"{kind}:{device_id}"
        self._device_id = device_id
        self._closed = False

        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id or f"dev-{self._device_id}-{uuid.uuid4().hex[:6]}",
        )

        # LWT: broker publishes this if our connection drops without a
        # clean disconnect. retained so any subscriber that connects later
        # sees the device as down.
        lwt_payload = json.dumps({
            "state": "down",
            "msg": "connection lost",
            "ts": time.time(),
        })
        self.client.will_set(
            f"device/{self._device_id}/state",
            lwt_payload,
            qos=1,
            retain=True,
        )

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        # Forward device state changes to the bus. Hook this BEFORE
        # connecting so we don't miss any transitions during startup.
        device.on_state_change(self._on_device_state_change)

        # Connect-or-warn: if the broker is unreachable, log and keep the
        # client running so paho's reconnect loop can recover.
        try:
            self.client.connect(self.broker_host, self.broker_port, keepalive=30)
        except Exception as ex:
            log.warning(
                "MQTTDeviceAdapter[%s]: initial connect to %s:%s failed (%s); "
                "will retry in background.",
                self._device_id, self.broker_host, self.broker_port, ex,
            )

        self.client.loop_start()

    # ── paho callbacks (v2 API) ───────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        # Republish info + current state on every connect so the broker
        # has fresh retained messages even after a disconnect/reconnect.
        try:
            self._publish_info()
            self._publish_state(self.device.state, self.device.msg)
            client.subscribe(f"device/{self._device_id}/cmd/+", qos=1)
            log.info("MQTTDeviceAdapter[%s]: connected, info+state published, cmd subscribed.",
                     self._device_id)
        except Exception:
            log.exception("MQTTDeviceAdapter[%s]: on_connect failed", self._device_id)

    def _on_disconnect(self, client, userdata, *args, **kwargs):
        # paho schedules its own reconnect; we just log.
        log.info("MQTTDeviceAdapter[%s]: disconnected (paho will retry)", self._device_id)

    def _on_message(self, client, userdata, message):
        try:
            self._dispatch_command(message.topic, message.payload)
        except Exception:
            log.exception("MQTTDeviceAdapter[%s]: command handler failed for %s",
                          self._device_id, getattr(message, "topic", "<unknown>"))

    # ── Device → MQTT ─────────────────────────────────────────────────────

    def _on_device_state_change(self, state: str, msg: str) -> None:
        """Camera (or any device) calls this on every transition."""
        self._publish_state(state, msg)

    def _publish_info(self) -> None:
        payload = json.dumps({
            "id": self._device_id,
            "kind": self.kind,
            "critical": self.critical,
            "meta": self.meta,
        })
        self.client.publish(
            f"device/{self._device_id}/info",
            payload,
            qos=1,
            retain=True,
        )

    def _publish_state(self, state: str, msg: str) -> None:
        payload = json.dumps({
            "state": state,
            "msg": msg or "",
            "ts": time.time(),
        })
        self.client.publish(
            f"device/{self._device_id}/state",
            payload,
            qos=1,
            retain=True,
        )

    # ── MQTT → Device ─────────────────────────────────────────────────────

    def _dispatch_command(self, topic: str, raw_payload: bytes) -> None:
        """Topic shape: ``device/<id>/cmd/<action>``."""
        parts = topic.split("/")
        if len(parts) < 4 or parts[2] != "cmd":
            return
        action = parts[3]
        try:
            req = json.loads(raw_payload.decode())
        except Exception:
            log.warning("MQTTDeviceAdapter[%s]: bad JSON on %s", self._device_id, topic)
            return
        req_id = req.get("req_id")

        # Recover/release may block (USB rebind, hardware reset). Run on a
        # worker thread so paho's network loop stays responsive.
        threading.Thread(
            target=self._run_command,
            args=(action, req_id),
            name=f"mqtt-cmd-{self._device_id}-{action}",
            daemon=True,
        ).start()

    def _run_command(self, action: str, req_id: Optional[str]) -> None:
        try:
            if action == "recover":
                ok = bool(self.device.recover())
            elif action == "release":
                self.device.release()
                ok = True
            else:
                log.warning("MQTTDeviceAdapter[%s]: unknown action %r",
                            self._device_id, action)
                ok = False
            reply = {
                "req_id": req_id,
                "ok": ok,
                "state": getattr(self.device, "state", "down"),
                "msg": getattr(self.device, "msg", ""),
            }
        except Exception as ex:
            log.exception("MQTTDeviceAdapter[%s]: %s raised", self._device_id, action)
            reply = {
                "req_id": req_id,
                "ok": False,
                "state": getattr(self.device, "state", "down"),
                "msg": f"{type(ex).__name__}: {ex}",
            }

        try:
            self.client.publish(
                f"device/{self._device_id}/cmd/{action}/reply",
                json.dumps(reply),
                qos=1,
                retain=False,
            )
        except Exception:
            log.exception("MQTTDeviceAdapter[%s]: reply publish failed for %s",
                          self._device_id, action)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Stop the network loop and disconnect cleanly. Safe to call twice.

        On clean disconnect the broker does NOT publish the LWT — that's
        intentional; if the device is being shut down on purpose, the
        last retained ``state`` message stands as the final word.
        """
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
