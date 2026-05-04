"""Compat shim — the canonical MQTTDeviceAdapter now lives in workspace.

The real implementation moved to :mod:`workspace.devices.adapter` so any
device service (camera, printer, robot, …) can import the same code from
one place. This module just re-exports it to keep existing imports
(``from dorna_vision.server.mqtt_adapter import MQTTDeviceAdapter``)
working until callers migrate. New code should import from
``workspace.devices`` directly.
"""

from workspace.devices.adapter import (  # noqa: F401  (re-export)
    MQTTDeviceAdapter,
    DEFAULT_BROKER_HOST,
    DEFAULT_BROKER_PORT,
)

__all__ = ["MQTTDeviceAdapter", "DEFAULT_BROKER_HOST", "DEFAULT_BROKER_PORT"]
