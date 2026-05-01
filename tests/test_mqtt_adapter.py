"""Tests for dorna_vision.server.mqtt_adapter.MQTTDeviceAdapter.

Uses a fake paho client (no broker required). Drives ``on_connect`` /
``on_message`` directly to simulate broker activity.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import pytest

# Patch the adapter module to use a fake client BEFORE importing.
import dorna_vision.server.mqtt_adapter as mqtt_adapter_module


# ── Fake paho client ──────────────────────────────────────────────────────


class FakeMessage:
    def __init__(self, topic: str, payload: bytes | dict[str, Any]):
        self.topic = topic
        self.payload = (
            json.dumps(payload).encode() if isinstance(payload, dict) else payload
        )


class FakeClient:
    """Replacement for paho.mqtt.client.Client in tests."""

    instances: list["FakeClient"] = []

    def __init__(self, *args, callback_api_version=None, client_id="", **kwargs):
        self.client_id = client_id
        self.on_connect: Callable | None = None
        self.on_message: Callable | None = None
        self.on_disconnect: Callable | None = None
        self.connected = False
        self.loop_started = False
        self.subscriptions: list = []
        self.published: list[tuple[str, str, int, bool]] = []
        self.disconnected = False
        self._lwt: tuple[str, str, int, bool] | None = None
        FakeClient.instances.append(self)

    def will_set(self, topic, payload, qos=0, retain=False):
        self._lwt = (topic, payload, qos, retain)

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

    def subscribe(self, topic, qos=0):
        self.subscriptions.append((topic, qos))

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append((topic, payload, qos, retain))

    # Test-side ──────────────────────────────────────────────────────────
    def fire_connect(self):
        if self.on_connect:
            self.on_connect(self, None, {}, 0)

    def fire_message(self, topic: str, payload):
        if self.on_message:
            self.on_message(self, None, FakeMessage(topic, payload))

    def by_topic(self, topic_substr: str) -> list[tuple]:
        return [p for p in self.published if topic_substr in p[0]]


# ── Fake Device-shaped object ─────────────────────────────────────────────


class FakeCamera:
    def __init__(self, id: str = "camera:test-1"):
        self.id = id
        self.state = "down"
        self.msg = "not connected"
        self._listeners: list[Callable[[str, str], None]] = []
        self.recover_calls = 0
        self.release_calls = 0

    def on_state_change(self, callback):
        self._listeners.append(callback)

    def transition(self, state: str, msg: str = ""):
        self.state = state
        self.msg = msg
        for cb in list(self._listeners):
            cb(state, msg)

    def recover(self):
        self.recover_calls += 1
        # Default: simulate successful recovery.
        self.transition("ok", "")
        return True

    def release(self):
        self.release_calls += 1


# ── Fixture: build adapter with FakeClient injected ───────────────────────


@pytest.fixture(autouse=True)
def _patch_paho_client(monkeypatch):
    """Force the adapter to use FakeClient instead of paho.mqtt.Client."""

    class _ClientFactoryShim:
        VERSION2 = object()

        def Client(self, *args, **kwargs):
            return FakeClient(*args, **kwargs)

    # Replace the mqtt module symbol the adapter references
    fake_mqtt = type("FakeMqttModule", (), {})()
    fake_mqtt.Client = lambda *a, **kw: FakeClient(*a, **kw)
    fake_mqtt.CallbackAPIVersion = type("V", (), {"VERSION2": object()})

    monkeypatch.setattr(mqtt_adapter_module, "mqtt", fake_mqtt)
    FakeClient.instances.clear()
    yield


def _make_adapter(device, **kwargs):
    return mqtt_adapter_module.MQTTDeviceAdapter(
        device,
        kind=kwargs.pop("kind", "camera"),
        critical=kwargs.pop("critical", True),
        meta=kwargs.pop("meta", {"sn": "test"}),
        broker_host="test-broker",
        broker_port=1883,
        **kwargs,
    )


# ── Tests ─────────────────────────────────────────────────────────────────


def test_lwt_set_before_connect():
    cam = FakeCamera("camera:abc")
    _make_adapter(cam)

    client = FakeClient.instances[-1]
    assert client._lwt is not None
    topic, payload, qos, retain = client._lwt
    assert topic == "device/camera:abc/state"
    body = json.loads(payload)
    assert body["state"] == "down"
    assert qos == 1 and retain is True


def test_on_connect_publishes_info_and_state_and_subscribes():
    cam = FakeCamera("camera:abc")
    cam.state = "ok"
    cam.msg = ""
    adapter = _make_adapter(cam, critical=True, meta={"model": "D405"})
    client = FakeClient.instances[-1]

    client.fire_connect()

    info = client.by_topic("/info")
    state = client.by_topic("/state")
    assert len(info) == 1
    assert len(state) == 1

    info_payload = json.loads(info[0][1])
    assert info_payload["id"] == "camera:abc"
    assert info_payload["kind"] == "camera"
    assert info_payload["critical"] is True
    assert info_payload["meta"] == {"model": "D405"}

    state_payload = json.loads(state[0][1])
    assert state_payload["state"] == "ok"

    cmd_subs = [s for s in client.subscriptions if "cmd" in s[0]]
    assert len(cmd_subs) == 1
    assert cmd_subs[0][0] == "device/camera:abc/cmd/+"


def test_state_change_publishes_state_transition():
    cam = FakeCamera("camera:abc")
    adapter = _make_adapter(cam)
    client = FakeClient.instances[-1]
    client.fire_connect()
    client.published.clear()

    cam.transition("down", "usb gone")

    state_msgs = client.by_topic("/state")
    assert len(state_msgs) == 1
    payload = json.loads(state_msgs[0][1])
    assert payload["state"] == "down"
    assert payload["msg"] == "usb gone"
    assert state_msgs[0][3] is True  # retained


def test_recover_command_invokes_device_and_replies(tmp_path):
    import time
    cam = FakeCamera("camera:abc")
    adapter = _make_adapter(cam)
    client = FakeClient.instances[-1]
    client.fire_connect()
    client.published.clear()

    client.fire_message(
        "device/camera:abc/cmd/recover",
        {"req_id": "r1"},
    )

    # The command runs on a worker thread — wait briefly.
    deadline = time.time() + 1.0
    while time.time() < deadline and cam.recover_calls == 0:
        time.sleep(0.01)

    assert cam.recover_calls == 1
    # Reply should be published on the spec'd topic.
    deadline = time.time() + 1.0
    replies = []
    while time.time() < deadline:
        replies = client.by_topic("/cmd/recover/reply")
        if replies:
            break
        time.sleep(0.01)
    assert len(replies) == 1
    body = json.loads(replies[0][1])
    assert body["req_id"] == "r1"
    assert body["ok"] is True
    # reply is NOT retained per spec
    assert replies[0][3] is False


def test_release_command_invokes_device_and_replies():
    import time
    cam = FakeCamera("camera:abc")
    adapter = _make_adapter(cam)
    client = FakeClient.instances[-1]
    client.fire_connect()

    client.fire_message(
        "device/camera:abc/cmd/release",
        {"req_id": "r2"},
    )

    deadline = time.time() + 1.0
    while time.time() < deadline and cam.release_calls == 0:
        time.sleep(0.01)
    assert cam.release_calls == 1


def test_unknown_command_replies_not_ok():
    import time
    cam = FakeCamera("camera:abc")
    adapter = _make_adapter(cam)
    client = FakeClient.instances[-1]
    client.fire_connect()
    client.published.clear()

    client.fire_message(
        "device/camera:abc/cmd/garbage",
        {"req_id": "r3"},
    )

    deadline = time.time() + 1.0
    replies = []
    while time.time() < deadline:
        replies = client.by_topic("/cmd/garbage/reply")
        if replies:
            break
        time.sleep(0.01)
    assert len(replies) == 1
    body = json.loads(replies[0][1])
    assert body["ok"] is False


def test_bad_json_payload_does_not_crash():
    cam = FakeCamera("camera:abc")
    adapter = _make_adapter(cam)
    client = FakeClient.instances[-1]
    client.fire_connect()

    # Should swallow silently.
    client.fire_message("device/camera:abc/cmd/recover", b"not json")
    assert cam.recover_calls == 0


def test_close_is_idempotent():
    cam = FakeCamera("camera:abc")
    adapter = _make_adapter(cam)
    client = FakeClient.instances[-1]
    adapter.close()
    adapter.close()
    assert client.disconnected is True
    assert client.loop_started is False


def test_device_without_id_raises():
    class NoId:
        id = ""
        state = "down"
        msg = ""
        def on_state_change(self, cb): pass
        def recover(self): return True
        def release(self): pass

    with pytest.raises(ValueError):
        _make_adapter(NoId())


def test_id_prefix_auto_normalized_when_missing():
    """Bare ids (no kind prefix) get auto-prefixed with the kind."""
    cam = FakeCamera("130322274110")  # bare serial, no "camera:" prefix
    adapter = _make_adapter(cam, kind="camera")
    client = FakeClient.instances[-1]
    client.fire_connect()

    info = client.by_topic("/info")
    assert len(info) == 1
    payload = json.loads(info[0][1])
    # Adapter normalized the id even though the device class didn't.
    assert payload["id"] == "camera:130322274110"
    assert info[0][0] == "device/camera:130322274110/info"


def test_id_prefix_kept_when_already_present():
    """Already-prefixed ids are left alone — no double-prefixing."""
    cam = FakeCamera("camera:abc")
    adapter = _make_adapter(cam, kind="camera")
    client = FakeClient.instances[-1]
    client.fire_connect()

    info = client.by_topic("/info")
    payload = json.loads(info[0][1])
    assert payload["id"] == "camera:abc"  # not "camera:camera:abc"


def test_reconnect_republishes_info_and_state():
    cam = FakeCamera("camera:abc")
    cam.state = "ok"
    adapter = _make_adapter(cam)
    client = FakeClient.instances[-1]

    client.fire_connect()
    info_count_1 = len(client.by_topic("/info"))
    state_count_1 = len(client.by_topic("/state"))
    assert info_count_1 == 1 and state_count_1 == 1

    client.fire_connect()  # reconnect
    info_count_2 = len(client.by_topic("/info"))
    state_count_2 = len(client.by_topic("/state"))
    assert info_count_2 == 2
    assert state_count_2 == 2
