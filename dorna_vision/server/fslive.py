"""fslive — a folder over a WebSocket, live; its bytes over HTTP, streamed.

ONE MODULE, TWO SERVERS. This file is byte-identical in
``workspace/gui/orchestrator/fslive.py`` (the orchestrator's project
folders: data / results / rec) and ``dorna_vision/server/fslive.py``
(the vision unit's captures folder). The two repos do not import each
other's server code, so it is kept as a twin: change one, copy it to
the other. Each server supplies only its roots (``FilesSocket.resolve``).

WHAT GOES WHERE, AND WHY
  * The WebSocket carries everything that is small and interactive:
    ``open`` a folder (its listing), ``mkdir``, ``delete`` — each
    answered on the socket — and LIVE CHANGES the server pushes: a
    file that appears, changes or disappears in the open folder
    arrives as that one entry, so the page never reloads a listing.
  * HTTP carries file BYTES — download, folder zip, image preview,
    upload. Bytes over a WebSocket would pass through JavaScript memory
    in chunks and lose the browser's own download manager and image
    cache; HTTP streaming is the fast, bounded path.

COST ON THE PI
  * Live changes are Linux inotify: the kernel reports changes on the
    event loop's file descriptor. No thread, no polling — an open
    folder that does not change costs nothing. One watch per folder
    that some page has open, shared between pages, dropped with the
    last. Bursts (a camera saving many frames) are coalesced
    (``DEBOUNCE_S``) and each changed name is stat-ed once.
  * Every filesystem walk or stat runs in a worker thread, never on the
    event loop: a large folder never stalls the server's other traffic.
  * Downloads are written in 64 KB chunks and wait for the socket
    (``flush``) before reading the next: memory stays at one chunk per
    download, whatever the file size.
  * A folder zip is produced in a worker thread into a small bounded
    queue (stored, not deflated — images and recordings do not shrink,
    and deflate costs CPU): memory stays bounded, the loop stays free.
  * Uploads stream to disk as the bytes arrive (``UploadHandler``,
    Tornado ``stream_request_body``): a large file is never held in
    memory, and it appears under its name only once complete.

No dependency beyond Tornado and the C library. Without inotify (not
Linux) the folder still works; it just does not update by itself.
"""
from __future__ import annotations

import asyncio
import ctypes
import ctypes.util
import errno
import json
import os
import struct
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import tornado.ioloop
import tornado.web
import tornado.websocket

DEBOUNCE_S = 0.15            # coalesce a burst of changes into one push
CHUNK = 1 << 16              # 64 KB — download / zip / upload granularity
ZIP_QUEUE = 16               # zip chunks in flight: at most ~1 MB buffered
MAX_BODY = 64 << 30          # an upload may be large; it streams, never buffered

# Filesystem work off the event loop. Small on purpose: these are short
# stat / scandir / unlink calls, and a Pi does not need many of them.
_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="fslive")


def _run(fn, *a):
    return asyncio.get_running_loop().run_in_executor(_POOL, fn, *a)


# ── The folder, as rows ──────────────────────────────────────────────

def safe_join(base: Path, rel: str) -> Path:
    """``base/rel`` resolved, refusing anything that escapes ``base``."""
    base = Path(base).resolve()
    target = (base / (rel or "").lstrip("/")).resolve()
    if target != base and base not in target.parents:
        raise ValueError("path is outside the folder")
    return target


def entry(p: Path, base: Path) -> Optional[dict]:
    """One row: what it is, how big, how old — or None if it is gone."""
    try:
        st = p.stat()
    except FileNotFoundError:
        return None
    is_dir = p.is_dir()
    return {"name": p.name, "path": str(p.relative_to(base)), "dir": is_dir,
            "size": 0 if is_dir else st.st_size, "mtime": st.st_mtime}


def scan(folder: Path, base: Path) -> List[dict]:
    """The folder's rows, folders first then newest first; hidden names
    skipped. ``os.scandir`` — one directory read, one stat per entry."""
    out = []
    rel = str(Path(folder).relative_to(base))
    pre = "" if rel == "." else rel + "/"          # row paths as strings: no Path per entry
    try:
        with os.scandir(folder) as it:
            for de in it:
                if de.name.startswith("."):
                    continue
                try:
                    st = de.stat()
                    is_dir = de.is_dir()
                except FileNotFoundError:
                    continue
                out.append({"name": de.name, "path": pre + de.name,
                            "dir": is_dir, "size": 0 if is_dir else st.st_size,
                            "mtime": st.st_mtime})
    except FileNotFoundError:
        return []             # an absent folder lists EMPTY: "nothing yet" is the honest answer
    out.sort(key=lambda e: (not e["dir"], -e["mtime"]))
    return out


# ── inotify, on the event loop ───────────────────────────────────────

_IN = dict(CLOSE_WRITE=0x008, ATTRIB=0x004, MOVED_FROM=0x040, MOVED_TO=0x080,
           CREATE=0x100, DELETE=0x200, DELETE_SELF=0x400, MOVE_SELF=0x800,
           IGNORED=0x8000, Q_OVERFLOW=0x4000, ONLYDIR=0x01000000, NONBLOCK=0o4000,
           CLOEXEC=0o2000000)
_MASK = (_IN["CLOSE_WRITE"] | _IN["ATTRIB"] | _IN["MOVED_FROM"] | _IN["MOVED_TO"]
         | _IN["CREATE"] | _IN["DELETE"] | _IN["DELETE_SELF"] | _IN["MOVE_SELF"]
         | _IN["ONLYDIR"])
_EVENT = struct.Struct("iIII")


class Watcher:
    """Folders watched for the pages that have them open.

    ``subscribe(folder, cb)`` → token; ``cb(names, gone)`` runs on the
    event loop after a quiet ``DEBOUNCE_S`` with the set of names that
    changed (``gone`` when the folder itself was deleted or moved).
    One inotify watch per folder, shared, removed with its last page."""

    _instance: Optional["Watcher"] = None

    @classmethod
    def get(cls) -> "Watcher":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.fd = -1
        self._libc = None
        self._wd: Dict[str, int] = {}                  # folder -> watch
        self._path: Dict[int, str] = {}                # watch -> folder
        self._subs: Dict[str, Dict[int, Callable]] = {}
        self._pending: Dict[str, Set[str]] = {}
        self._gone: Set[str] = set()
        self._timer = None
        self._next = 0
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
            fd = libc.inotify_init1(_IN["NONBLOCK"] | _IN["CLOEXEC"])
            if fd < 0:
                raise OSError(ctypes.get_errno(), "inotify_init1")
            self._libc, self.fd = libc, fd
            tornado.ioloop.IOLoop.current().add_handler(fd, self._on_readable,
                                                        tornado.ioloop.IOLoop.READ)
        except Exception:
            self.fd = -1               # no live updates; listings still work

    @property
    def live(self) -> bool:
        return self.fd >= 0

    def subscribe(self, folder: Path, cb: Callable) -> Optional[int]:
        if not self.live:
            return None
        key = str(folder)
        if key not in self._wd:
            wd = self._libc.inotify_add_watch(self.fd, key.encode(), _MASK)
            if wd < 0:
                return None            # absent folder, or out of watches: no live view
            self._wd[key] = wd
            self._path[wd] = key
        self._next += 1
        self._subs.setdefault(key, {})[self._next] = cb
        return self._next

    def unsubscribe(self, folder: Path, token: Optional[int]) -> None:
        key = str(folder)
        subs = self._subs.get(key)
        if not subs or token not in subs:
            return
        del subs[token]
        if not subs:
            del self._subs[key]
            wd = self._wd.pop(key, None)
            if wd is not None:
                self._path.pop(wd, None)
                self._libc.inotify_rm_watch(self.fd, wd)
            self._pending.pop(key, None)

    def _on_readable(self, fd, events):
        while True:
            try:
                buf = os.read(fd, 64 * 1024)
            except BlockingIOError:
                break
            except OSError as ex:
                if ex.errno == errno.EINTR:
                    continue
                break
            if not buf:
                break
            i = 0
            while i + _EVENT.size <= len(buf):
                wd, mask, _cookie, n = _EVENT.unpack_from(buf, i)
                name = buf[i + _EVENT.size:i + _EVENT.size + n].split(b"\0", 1)[0]
                i += _EVENT.size + n
                if mask & _IN["Q_OVERFLOW"]:
                    for key in self._subs:            # lost events: resend the whole folder
                        self._pending.setdefault(key, set()).add("*")
                    continue
                key = self._path.get(wd)
                if key is None:
                    continue
                if mask & (_IN["DELETE_SELF"] | _IN["MOVE_SELF"] | _IN["IGNORED"]):
                    self._gone.add(key)
                elif name:
                    self._pending.setdefault(key, set()).add(name.decode(errors="surrogateescape"))
        if (self._pending or self._gone) and self._timer is None:
            self._timer = tornado.ioloop.IOLoop.current().call_later(DEBOUNCE_S, self._fire)

    def _fire(self):
        self._timer = None
        pending, self._pending = self._pending, {}
        gone, self._gone = self._gone, set()
        for key in gone:
            wd = self._wd.pop(key, None)
            if wd is not None:
                self._path.pop(wd, None)
            for cb in list(self._subs.pop(key, {}).values()):
                cb(set(), True)
            pending.pop(key, None)
        for key, names in pending.items():
            for cb in list(self._subs.get(key, {}).values()):
                cb(set(names), False)


# ── Bytes out: a file, a folder as a zip ─────────────────────────────

async def send_file(handler: tornado.web.RequestHandler, path: Path, *,
                    inline: bool = False, ctype: Optional[str] = None) -> None:
    """Stream ``path`` in CHUNK pieces, each flushed before the next is
    read: one chunk of memory per download."""
    handler.set_header("Content-Type", ctype or "application/octet-stream")
    if not inline:
        handler.set_header("Content-Disposition", f'attachment; filename="{path.name}"')
    size = (await _run(os.path.getsize, path))
    handler.set_header("Content-Length", str(size))
    fp = await _run(open, path, "rb")
    try:
        while True:
            chunk = await _run(fp.read, CHUNK)
            if not chunk:
                break
            handler.write(chunk)
            await handler.flush()
    finally:
        fp.close()


class _ZipSink:
    """Non-seekable sink for ``zipfile``: full CHUNKs go to ``put`` (which
    blocks the producer thread while the consumer is behind)."""
    def __init__(self, put):
        self.put, self.pos, self.buf = put, 0, bytearray()

    def write(self, b):
        self.buf += b
        self.pos += len(b)
        while len(self.buf) >= CHUNK:
            self.put(bytes(self.buf[:CHUNK]))
            del self.buf[:CHUNK]
        return len(b)

    def tell(self):
        return self.pos

    def flush(self):
        pass

    def close_out(self):
        if self.buf:
            self.put(bytes(self.buf))
            self.buf.clear()


async def send_zip(handler: tornado.web.RequestHandler, folder: Path, name: str) -> None:
    """Stream ``folder`` as ``name.zip`` (the folder at the top of the
    archive). Built in its own thread into a bounded asyncio queue (the
    thread waits when it is full); the handler drains it to the socket.
    Stored, not deflated."""
    handler.set_header("Content-Type", "application/zip")
    handler.set_header("Content-Disposition", f'attachment; filename="{name}.zip"')
    loop = asyncio.get_running_loop()
    aq: "asyncio.Queue" = asyncio.Queue(maxsize=ZIP_QUEUE)
    DONE = object()
    stop = threading.Event()

    def put(item):
        if stop.is_set():
            raise _Abort()
        asyncio.run_coroutine_threadsafe(aq.put(item), loop).result()

    def produce():
        try:
            sink = _ZipSink(put)
            with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
                for dirpath, dirnames, filenames in os.walk(folder):
                    dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
                    for fn in sorted(filenames):
                        if stop.is_set():
                            return
                        if fn.startswith("."):
                            continue
                        p = Path(dirpath) / fn
                        try:
                            zf.write(p, arcname=str(Path(name) / p.relative_to(folder)))
                        except FileNotFoundError:
                            continue           # vanished while zipping
            sink.close_out()
        except _Abort:
            return
        except Exception as ex:                # surfaces as a truncated download
            loop.call_soon_threadsafe(aq.put_nowait, ex)
        finally:
            if not stop.is_set():
                asyncio.run_coroutine_threadsafe(aq.put(DONE), loop)

    threading.Thread(target=produce, name="fslive-zip", daemon=True).start()
    try:
        while True:
            item = await aq.get()
            if item is DONE:
                break
            if isinstance(item, Exception):
                raise item
            handler.write(item)
            await handler.flush()
    finally:
        stop.set()                             # the download ended or was cancelled
        while not aq.empty():                  # release a producer blocked on a full queue
            aq.get_nowait()


class _Abort(Exception):
    """The download went away: the zip producer stops."""


# ── Bytes in: an upload, streamed to disk ────────────────────────────

@tornado.web.stream_request_body
class UploadHandler(tornado.web.RequestHandler):
    """``PUT <upload url>?path=<folder>&name=<file>`` with the file as the
    raw body. The bytes go to ``.<name>.part`` in the folder as they
    arrive and the file takes its name when complete — a half upload
    never looks like a file. Subclasses supply ``resolve(root) -> base``
    (and may override ``created``)."""

    def resolve(self, *args) -> Path:
        raise NotImplementedError

    def created(self, path: Path) -> None:
        """Called for every file / folder the upload created."""

    def authorized(self) -> bool:
        return True

    def prepare(self):
        self._fp = None
        self._err = None
        self.request.connection.set_max_body_size(MAX_BODY)
        try:
            if not self.authorized():
                raise PermissionError("Unauthorized")
            if self.request.method != "PUT":
                raise ValueError("upload with PUT")
            base = self.resolve(*self.path_args)
            folder = safe_join(base, self.get_argument("path", ""))
            name = os.path.basename(self.get_argument("name", ""))
            if not name or name.startswith("."):
                raise ValueError("a file name is required")
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)
                self.created(folder)
            self._base = base
            self._dest = safe_join(folder, name)
            self._part = folder / f".{name}.part"
            self._fp = open(self._part, "wb")
        except Exception as ex:
            self._err = ex

    def data_received(self, chunk: bytes):
        if self._fp is not None:
            self._fp.write(chunk)

    def put(self, *args):
        if self._err is not None:
            self.set_status(401 if isinstance(self._err, PermissionError) else 400)
            self.write({"error": str(self._err)})
            return
        self._fp.close()
        self._fp = None
        os.replace(self._part, self._dest)
        self.created(self._dest)
        self.write({"ok": True, "path": str(self._dest.relative_to(self._base)),
                    "name": self._dest.name, "abs": str(self._dest)})

    def on_connection_close(self):
        # The browser went away mid-upload: no half file left behind.
        if self._fp is not None:
            try:
                self._fp.close()
                os.unlink(self._part)
            except OSError:
                pass
            self._fp = None


# ── The folder socket ────────────────────────────────────────────────

class FilesSocket(tornado.websocket.WebSocketHandler):
    """One page's view of one folder at a time.

    Client → server (each answered ``{"id", "ok": true, ...}`` or
    ``{"id", "ok": false, "error"}``):
        {"id", "op": "open",   "root", "path"}  → {"folder": {abs, path, entries, ...}}
        {"id", "op": "mkdir",  "root", "path"}  → {}
        {"id", "op": "delete", "root", "path"}  → {}   (a file, or an EMPTY folder)
    Server → client, for the folder last opened:
        {"ev": "change", "root", "path", "upsert": [row, ...], "remove": [path, ...]}
        {"ev": "gone",   "root", "path"}        the folder itself was deleted

    Subclasses supply ``resolve(root) -> base Path``, and may add
    ``meta(root, base) -> dict`` (merged into the ``open`` answer),
    ``created(path)`` and ``authorized()``."""

    def resolve(self, root: str) -> Path:
        raise NotImplementedError

    def meta(self, root: str, base: Path) -> dict:
        return {}

    def created(self, path: Path) -> None:
        """Called for every file / folder a request created."""

    def authorized(self) -> bool:
        return True

    def open(self, *args):
        self._view: Optional[Tuple[str, str, Path, Path]] = None   # root, rel, base, folder
        self._token = None
        if not self.authorized():
            self.close(4401, "Unauthorized")

    def on_close(self):
        self._unwatch()

    def _unwatch(self):
        if self._view is not None:
            Watcher.get().unsubscribe(self._view[3], self._token)
        self._view, self._token = None, None

    def _send(self, msg: dict):
        try:
            self.write_message(json.dumps(msg))
        except tornado.websocket.WebSocketClosedError:
            self._unwatch()

    async def on_message(self, raw):
        try:
            msg = json.loads(raw)
        except ValueError:
            return
        mid = msg.get("id")
        try:
            out = await self._op(msg.get("op"), msg.get("root") or "", msg.get("path") or "")
            self._send({"id": mid, "ok": True, **(out or {})})
        except Exception as ex:
            self._send({"id": mid, "ok": False, "error": str(ex)})

    async def _op(self, op, root, rel):
        base = Path(self.resolve(root)).resolve()
        target = safe_join(base, rel)
        if op == "open":
            entries = await _run(scan, target, base)
            self._unwatch()
            self._view = (root, rel, base, target)
            token = Watcher.get().subscribe(target, self._on_change)
            self._token = token
            return {"folder": {"root": root, "path": rel, "abs": str(target),
                               "entries": entries, "live": token is not None,
                               **self.meta(root, base)}}
        if op == "mkdir":
            if not rel:
                raise ValueError("a name is required")
            await _run(lambda: target.mkdir(parents=True, exist_ok=True))
            self.created(target)
            return {}
        if op == "delete":
            if target == base:
                raise ValueError("cannot delete the folder itself")

            def rm():
                if not target.exists():
                    raise ValueError("no such file")
                if target.is_dir():
                    if any(target.iterdir()):
                        raise ValueError("folder is not empty — empty it first")
                    target.rmdir()
                else:
                    target.unlink()
            await _run(rm)
            return {}
        raise ValueError(f"unknown op: {op}")

    def _on_change(self, names: Set[str], gone: bool):
        view = self._view
        if view is None:
            return
        root, rel, base, folder = view
        if gone:
            self._view, self._token = None, None       # the watcher dropped it
            self._send({"ev": "gone", "root": root, "path": rel})
            return

        def rows():
            if "*" in names:                           # overflow: the whole folder again
                return scan(folder, base), None
            up, rm = [], []
            for n in sorted(names):
                if n.startswith("."):
                    continue
                e = entry(folder / n, base)
                if e is None:
                    rm.append(str((folder / n).relative_to(base)))
                else:
                    up.append(e)
            return up, rm

        async def push():
            up, rm = await _run(rows)
            if self._view is not view:
                return                                 # the page moved on meanwhile
            if rm is None:
                self._send({"ev": "change", "root": root, "path": rel, "reset": True, "upsert": up, "remove": []})
            elif up or rm:
                self._send({"ev": "change", "root": root, "path": rel, "upsert": up, "remove": rm})
        tornado.ioloop.IOLoop.current().spawn_callback(push)
