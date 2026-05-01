"""Tests for dorna_vision.server.mqtt_relay.MQTTDeviceObserver.

Uses a fake paho client (no broker required). Same pattern as
test_mqtt_adapter.py.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import pytest

import dorna_vision.server.mqtt_relay as relay_module


class FakeMessage:
    def __init__(self, topic: str, payload):
        self.topic = topic
        self.payload = json.dumps(payload).encode() if isinstance(payload, dict) else payload


class FakeClient:
    instances: list["FakeClient"] = []

    def __init__(self, *args, callback_api_version=None, client_id="", **kwargs):
        self.client_id = client_id
        self.on_connect = None
        self.on_message = None
        self.on_disconnect = None
        self.connected = False
        self.loop_started = False
        self.subscriptions: list = []
        self.disconnected = False
        FakeClient.instances.append(self)

    def connect(self, host, port, keepalive=30):
        self.host = host
        self.port = port
        self.connected = True

    def loop_start(self):
        self.loop_started = True

    def loop_stop(self):
        self.loop_started = False

    def disconnect(self):
        self.disconnected = True
        self.connected = False

    def subscribe(self, topics):
        if isinstance(topics, list):
            self.subscriptions.extend(topics)
        else:
            self.subscriptions.append(topics)

    def fire_connect(self):
        if self.on_connect:
            self.on_connect(self, None, {}, 0)

    def fire_message(self, topic: str, payload):
        if self.on_message:
            self.on_message(self, None, FakeMessage(topic, payload))


@pytest.fixture(autouse=True)
def _patch_paho_client(monkeypatch):
    fake_mqtt = type("FakeMqttModule", (), {})()
    fake_mqtt.Client = lambda *a, **kw: FakeClient(*a, **kw)
    fake_mqtt.CallbackAPIVersion = type("V", (), {"VERSION2": object()})
    monkeypatch.setattr(relay_module, "mqtt", fake_mqtt)
    FakeClient.instances.clear()
    yield


def _make_observer():
    received: list[dict] = []
    obs = relay_module.MQTTDeviceObserver(
        broadcast=received.append,
        broker_host="test",
        broker_port=1883,
    )
    client = FakeClient.instances[-1]
    client.fire_connect()
    return obs, client, received


def test_subscribes_to_info_and_state_on_connect():
    obs, client, _ = _make_observer()
    topics = [s[0] for s in client.subscriptions]
    assert "device/+/info" in topics
    assert "device/+/state" in topics


def test_state_message_broadcasts_event_and_caches_entry():
    obs, client, received = _make_observer()
    client.fire_message(
        "device/camera:abc/state",
        {"state": "down", "msg": "lost", "ts": 100.0},
    )
    assert len(received) == 1
    evt = received[0]
    assert evt["type"] == "device_state"
    assert evt["id"] == "camera:abc"
    assert evt["serial_number"] == "abc"
    assert evt["state"] == "down"
    assert evt["msg"] == "lost"

    snap = obs.snapshot()
    assert any(s["id"] == "camera:abc" and s["state"] == "down" for s in snap)


def test_info_message_updates_kind_and_critical_in_cache():
    obs, client, received = _make_observer()
    client.fire_message(
        "device/camera:abc/info",
        {"id": "camera:abc", "kind": "camera", "critical": True, "meta": {"model": "D405"}},
    )
    assert len(received) == 1
    assert received[0]["type"] == "device_info"
    assert received[0]["kind"] == "camera"
    assert received[0]["critical"] is True
    assert received[0]["meta"] == {"model": "D405"}


def test_serial_extraction_with_and_without_prefix():
    obs, client, received = _make_observer()
    client.fire_message("device/camera:130322274110/state", {"state": "ok", "msg": "", "ts": 1.0})
    client.fire_message("device/raw_id/state", {"state": "ok", "msg": "", "ts": 1.0})
    assert received[0]["serial_number"] == "130322274110"
    assert received[1]["serial_number"] == "raw_id"


def test_bad_json_swallowed():
    obs, client, received = _make_observer()
    client.fire_message("device/camera:abc/state", b"not json")
    assert received == []


def test_unrelated_topics_ignored():
    obs, client, received = _make_observer()
    client.fire_message("not/a/device/topic", {"state": "ok"})
    client.fire_message("device/abc/cmd/recover", {"req_id": "x"})
    client.fire_message("device/abc/cmd/recover/reply", {"req_id": "x", "ok": True})
    assert received == []


def test_snapshot_returns_all_known_devices():
    obs, client, _ = _make_observer()
    client.fire_message(
        "device/camera:abc/info",
        {"id": "camera:abc", "kind": "camera", "critical": True, "meta": {}},
    )
    client.fire_message("device/camera:abc/state", {"state": "ok", "msg": "", "ts": 1.0})
    client.fire_message(
        "device/printer:p1/info",
        {"id": "printer:p1", "kind": "printer", "critical": False, "meta": {}},
    )
    client.fire_message("device/printer:p1/state", {"state": "down", "msg": "no paper", "ts": 2.0})

    snap = obs.snapshot()
    assert len(snap) == 2
    ids = {s["id"] for s in snap}
    assert ids == {"camera:abc", "printer:p1"}


def test_broadcast_listener_failure_does_not_break_observer():
    def bad(_evt):
        raise RuntimeError("boom")

    obs = relay_module.MQTTDeviceObserver(
        broadcast=bad, broker_host="test", broker_port=1883,
    )
    client = FakeClient.instances[-1]
    client.fire_connect()
    # Should not raise:
    client.fire_message("device/camera:x/state", {"state": "ok", "msg": "", "ts": 1.0})
    # Cache still updates so other subscribers are unaffected.
    snap = obs.snapshot()
    assert len(snap) == 1


def test_close_is_idempotent():
    obs, client, _ = _make_observer()
    obs.close()
    obs.close()
    assert client.disconnected is True
