import asyncio
import json
import traceback
from concurrent.futures import ThreadPoolExecutor

import tornado.websocket

from .handlers import HANDLERS, CAMERA_BOUND, BINARY_INBOUND, _to_jsonable
from .session import ClientSession


class VisionWSHandler(tornado.websocket.WebSocketHandler):
    """
    JSON-over-WebSocket protocol:

        client -> server: {"cmd": "<name>", "id": <int>, "args": {...}}
        server -> client: {"id": <int>, "stat": 2, ...result}          on success
                          {"id": <int>, "error": {"code":..,"msg":..}} on failure

    Binary replies (e.g. images) are sent as:
        1) a JSON envelope with "binary_follows": true
        2) a raw binary websocket frame immediately after

    Binary-bound commands (like detection_run) are dispatched onto the camera's
    single-worker executor so they serialize per camera, while different cameras
    run in parallel. Non-camera commands run on a shared thread pool.
    """

    # raise the default 10 MB cap so we can comfortably ship JPEGs
    max_message_size = 32 * 1024 * 1024

    def check_origin(self, origin):
        return True

    def initialize(self, camera_pool, robot_pool, default_executor):
        self.camera_pool = camera_pool
        self.robot_pool = robot_pool
        self.default_executor = default_executor
        self.session = None
        self._write_lock = asyncio.Lock()
        # When a JSON envelope arrives with binary_follows=true and its cmd
        # is in BINARY_INBOUND, we stash {handler, cmd, msg_id, args} here
        # and wait for the next binary frame to pair them up.
        self._pending_inbound = None

    async def open(self):
        self.session = ClientSession(self.camera_pool, self.robot_pool)
        try:
            peer = "%s:%s" % self.request.connection.context.address
        except Exception:
            peer = "?"
        print("client connected: %s" % peer)

    def on_close(self):
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                traceback.print_exc()
            self.session = None
        print("client disconnected (code=%s, reason=%s)" % (self.close_code, self.close_reason))

    async def on_message(self, message):
        # Binary frame: must follow a JSON envelope that asked for one.
        if isinstance(message, (bytes, bytearray, memoryview)):
            pending = self._pending_inbound
            self._pending_inbound = None
            if pending is None:
                # Unexpected binary frame — drop it. (Could log/warn but no
                # client should be sending bytes without a paired envelope.)
                return
            handler, cmd, msg_id, args = pending
            asyncio.create_task(self._dispatch(handler, cmd, msg_id, args, bytes(message)))
            return

        try:
            payload = json.loads(message)
        except Exception as ex:
            await self._send_json({"error": {"code": "BAD_ARGS", "msg": "invalid json: %s" % ex}})
            return

        msg_id = payload.get("id")
        cmd = payload.get("cmd")
        args = payload.get("args", {}) or {}

        if not cmd:
            await self._send_json({"id": msg_id, "error": {"code": "BAD_ARGS", "msg": "missing cmd"}})
            return

        handler = HANDLERS.get(cmd)
        if handler is None:
            await self._send_json({"id": msg_id, "error": {"code": "BAD_ARGS", "msg": "unknown cmd: %s" % cmd}})
            return

        # Binary-capable commands (detection_add, call) wait for a follow-up
        # binary frame ONLY if the envelope flagged binary_follows=true.
        # Otherwise dispatch immediately with data=None so the handler
        # can fall back to a path-based code path.
        if payload.get("binary_follows") and cmd in BINARY_INBOUND:
            if self._pending_inbound is not None:
                prev_id = self._pending_inbound[2]
                self._pending_inbound = None
                await self._send_json({"id": prev_id, "error": {"code": "PROTOCOL", "msg": "upload aborted by another upload"}})
            self._pending_inbound = (handler, cmd, msg_id, args)
            return

        asyncio.create_task(self._dispatch(handler, cmd, msg_id, args))

    async def _dispatch(self, handler, cmd, msg_id, args, data=None):
        loop = asyncio.get_running_loop()
        executor = self._resolve_executor(cmd, args)
        try:
            if cmd in BINARY_INBOUND:
                # Always pass data (None when no binary follow), so the
                # handler can branch on its presence without needing two
                # signatures.
                result = await loop.run_in_executor(executor, handler, self.session, args, data)
            else:
                result = await loop.run_in_executor(executor, handler, self.session, args)
        except Exception as ex:
            code = _error_code(ex)
            traceback.print_exc()
            await self._send_json({"id": msg_id, "error": {"code": code, "msg": str(ex)}})
            return

        await self._send_result(msg_id, result)

    def _resolve_executor(self, cmd, args):
        if cmd in CAMERA_BOUND:
            # camera_get_img is keyed by serial_number directly; the rest are keyed
            # by detection name and resolve via the session.
            serial_number = args.get("serial_number") if cmd == "camera_get_img" else None
            if serial_number is None:
                name = args.get("name")
                serial_number = self.session.detection_camera_serial_number(name) if name else None
            if serial_number is not None:
                ex = self.camera_pool.executor(serial_number)
                if ex is not None:
                    return ex
        elif cmd == "call":
            target = args.get("target")
            name = args.get("name")
            serial_number = None
            if target == "detection" and name:
                serial_number = self.session.detection_camera_serial_number(name)
            elif target == "camera" and name:
                serial_number = name
            if serial_number is not None:
                ex = self.camera_pool.executor(serial_number)
                if ex is not None:
                    return ex
        return self.default_executor

    async def _send_result(self, msg_id, result):
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], (bytes, bytearray, memoryview)):
            meta, binary = result
            envelope = {"id": msg_id, "stat": 2, "binary_follows": True}
            if isinstance(meta, dict):
                envelope.update(_to_jsonable(meta))
            async with self._write_lock:
                await self._write_json(envelope)
                await self._write_binary(bytes(binary))
            return

        envelope = {"id": msg_id, "stat": 2}
        if isinstance(result, dict):
            envelope.update(_to_jsonable(result))
        else:
            envelope["result"] = _to_jsonable(result)
        await self._send_json(envelope)

    async def _send_json(self, envelope):
        async with self._write_lock:
            await self._write_json(envelope)

    async def _write_json(self, envelope):
        try:
            await self.write_message(json.dumps(envelope))
        except tornado.websocket.WebSocketClosedError:
            pass

    async def _write_binary(self, data):
        try:
            await self.write_message(data, binary=True)
        except tornado.websocket.WebSocketClosedError:
            pass


def _error_code(ex):
    name = type(ex).__name__
    msg = str(ex).lower()
    if isinstance(ex, ValueError):
        if "not found" in msg:
            return "NOT_FOUND"
        return "BAD_ARGS"
    if isinstance(ex, KeyError):
        return "NOT_FOUND"
    if "camera" in msg:
        return "CAMERA_FAIL"
    if "robot" in msg:
        return "ROBOT_FAIL"
    return "INTERNAL"
