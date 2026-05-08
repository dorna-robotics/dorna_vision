"""Compat shim — the canonical MQTTDeviceAdapter lives in dorna_devices.

The real implementation lives in :mod:`dorna_devices.adapter`, the small
standalone package that the vision Pi (and any other device service) can
install without dragging in the full workspace SDK. This module
re-exports it so existing imports
(``from dorna_vision.server.mqtt_adapter import MQTTDeviceAdapter``)
keep working. New code should import from ``dorna_devices`` directly.
"""

from dorna_devices.adapter import (  # noqa: F401  (re-export)
    MQTTDeviceAdapter,
    DEFAULT_BROKER_HOST,
    DEFAULT_BROKER_PORT,
)

__all__ = ["MQTTDeviceAdapter", "DEFAULT_BROKER_HOST", "DEFAULT_BROKER_PORT"]
