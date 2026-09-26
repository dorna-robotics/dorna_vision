import asyncio
import os
import signal
from concurrent.futures import ThreadPoolExecutor

import tornado.ioloop
import tornado.web

from .pools import CameraPool, RobotPool
from .ws_handler import VisionWSHandler
from .mqtt_relay import MQTTDeviceObserver
from .files import FilesHandler, FilesSocket, UploadHandler, default_captures_dir, ensure_root


DEFAULT_PORT = 80

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
STATIC_DIR = os.path.join(WEB_DIR, "static")


class IndexHandler(tornado.web.RequestHandler):
    """Serve web/index.html at the root."""
    def get(self):
        self.set_header("Cache-Control", "no-store")
        with open(os.path.join(WEB_DIR, "index.html"), "rb") as fh:
            self.write(fh.read())


class NoCacheStaticFileHandler(tornado.web.StaticFileHandler):
    """Static files served with no-cache headers — keeps the browser
    from holding onto stale JS during dev. Negligible overhead at our
    scale (a handful of small files)."""
    def set_extra_headers(self, path):
        self.set_header("Cache-Control", "no-store")


# WebSocket keepalive. Without server pings an idle GUI connection carries
# no traffic at all, and anything stateful between the browser and this
# unit — the GL.iNet's NAT table (the GUI is usually reached from its
# Wi-Fi side), a sleeping laptop, a browser throttling a background tab —
# quietly forgets it. Neither end notices until the next request, which
# then just times out; the GUI showed "disconnected" until a page reload.
# A ping every 20 s keeps every hop's idle timer reset, and a peer that
# stops answering pongs is reaped after 60 s instead of holding a session
# open forever.
WS_PING_INTERVAL_S = 20
WS_PING_TIMEOUT_S = 60


def make_app(camera_pool, robot_pool, default_executor, captures_dir):
    return tornado.web.Application([
        (r"/ws", VisionWSHandler, {
            "camera_pool": camera_pool,
            "robot_pool": robot_pool,
            "default_executor": default_executor,
        }),
        (r"/static/(.*)", NoCacheStaticFileHandler, {"path": STATIC_DIR}),
        # Files page — the captures folder (see files.py). Before the SPA
        # fallback, which would otherwise answer these with index.html.
        (r"/api/files", FilesHandler, {"root": captures_dir}),
        (r"/api/files/upload", UploadHandler, {"root": captures_dir}),
        (r"/ws/files", FilesSocket, {"root": captures_dir}),
        # SPA fallback: any path that isn't /ws or /static/* serves index.html
        # so /, /cameras, /robots, /playground all render the same shell and
        # the client-side router picks the section.
        (r"/.*", IndexHandler),
    ],
        websocket_ping_interval=WS_PING_INTERVAL_S,
        websocket_ping_timeout=WS_PING_TIMEOUT_S,
    )


async def run_server(host="0.0.0.0", port=DEFAULT_PORT, max_workers=8,
                     mqtt_enabled=True, mqtt_broker_host=None, mqtt_broker_port=None,
                     captures_dir=None):
    # Capture the running loop so cross-thread broadcasts can schedule on it.
    VisionWSHandler.set_ioloop(asyncio.get_running_loop())

    camera_pool = CameraPool(
        mqtt_enabled=mqtt_enabled,
        mqtt_broker_host=mqtt_broker_host,
        mqtt_broker_port=mqtt_broker_port,
    )
    robot_pool = RobotPool()
    default_executor = ThreadPoolExecutor(max_workers=max_workers)

    # Web UI feed: subscribe to the same MQTT bus the orchestrator reads
    # from. Single source of truth — every consumer (this server's UI,
    # workspace orchestrator UI, future dashboards) sees identical data.
    # Falls back gracefully if the broker is unreachable: the local UI
    # will go blind for now and recover when the broker comes back.
    observer = None
    if mqtt_enabled:
        observer = MQTTDeviceObserver(
            broadcast=VisionWSHandler.broadcast,
            broker_host=mqtt_broker_host,
            broker_port=mqtt_broker_port,
        )
        # Reachable by the bus_connect handler (site-bus handshake).
        VisionWSHandler.bus_observer = observer
    VisionWSHandler.set_observer(observer)

    captures = ensure_root(captures_dir or default_captures_dir())
    app = make_app(camera_pool, robot_pool, default_executor, captures)
    try:
        app.listen(port, address=host)
    except OSError as ex:
        print("failed to bind http://%s:%d/: %s" % (host, port, ex))
        print("another server may already be running on that port.")
        default_executor.shutdown(wait=False)
        return
    print("dorna_vision server listening:")
    print("  GUI:       http://%s:%d/" % (host, port))
    print("  WebSocket: ws://%s:%d/ws" % (host, port))
    print("  Files:     %s" % captures)

    stop_event = asyncio.Event()

    def _shutdown(*_):
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            # windows: fall back to default KeyboardInterrupt handling
            signal.signal(sig, lambda *_: _shutdown())

    try:
        await stop_event.wait()
    finally:
        print("shutting down...")
        default_executor.shutdown(wait=True)
        camera_pool.shutdown()
        robot_pool.shutdown()
        if observer is not None:
            try:
                observer.close()
            except Exception:
                pass


def main(host="0.0.0.0", port=DEFAULT_PORT, max_workers=8,
         mqtt_enabled=True, mqtt_broker_host=None, mqtt_broker_port=None,
         captures_dir=None):
    try:
        asyncio.run(run_server(
            host=host, port=port, max_workers=max_workers,
            mqtt_enabled=mqtt_enabled,
            mqtt_broker_host=mqtt_broker_host,
            mqtt_broker_port=mqtt_broker_port,
            captures_dir=captures_dir,
        ))
    except KeyboardInterrupt:
        pass
