import asyncio
import os
import signal
from concurrent.futures import ThreadPoolExecutor

import tornado.ioloop
import tornado.web

from .pools import CameraPool, RobotPool
from .ws_handler import VisionWSHandler


DEFAULT_PORT = 80

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
STATIC_DIR = os.path.join(WEB_DIR, "static")


class IndexHandler(tornado.web.RequestHandler):
    """Serve web/index.html at the root."""
    def get(self):
        self.set_header("Cache-Control", "no-store")
        with open(os.path.join(WEB_DIR, "index.html"), "rb") as fh:
            self.write(fh.read())


def make_app(camera_pool, robot_pool, default_executor):
    return tornado.web.Application([
        (r"/ws", VisionWSHandler, {
            "camera_pool": camera_pool,
            "robot_pool": robot_pool,
            "default_executor": default_executor,
        }),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": STATIC_DIR}),
        # SPA fallback: any path that isn't /ws or /static/* serves index.html
        # so /, /cameras, /robots, /playground all render the same shell and
        # the client-side router picks the section.
        (r"/.*", IndexHandler),
    ])


async def run_server(host="0.0.0.0", port=DEFAULT_PORT, max_workers=8):
    camera_pool = CameraPool()
    robot_pool = RobotPool()
    default_executor = ThreadPoolExecutor(max_workers=max_workers)

    app = make_app(camera_pool, robot_pool, default_executor)
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


def main(host="0.0.0.0", port=DEFAULT_PORT, max_workers=8):
    try:
        asyncio.run(run_server(host=host, port=port, max_workers=max_workers))
    except KeyboardInterrupt:
        pass
