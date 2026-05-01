import argparse

from .app import main, DEFAULT_PORT


def cli():
    parser = argparse.ArgumentParser(prog="dorna_vision.server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--mqtt-broker", default=None,
                        help="MQTT broker host (overrides $DEVICE_MQTT_HOST). "
                             "Default: localhost.")
    parser.add_argument("--mqtt-port", type=int, default=None,
                        help="MQTT broker port (overrides $DEVICE_MQTT_PORT). "
                             "Default: 1883.")
    parser.add_argument("--no-mqtt", action="store_true",
                        help="Disable MQTT health publishing entirely.")
    args = parser.parse_args()
    main(
        host=args.host,
        port=args.port,
        max_workers=args.max_workers,
        mqtt_enabled=not args.no_mqtt,
        mqtt_broker_host=args.mqtt_broker,
        mqtt_broker_port=args.mqtt_port,
    )


if __name__ == "__main__":
    cli()
