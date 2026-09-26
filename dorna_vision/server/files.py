"""files.py — the Files page backend: browse the captures folder.

One root, the captures folder (``~/captures`` of the user who started
the server, ``--captures`` overrides). Everything the Files page does is
here, all confined to that root:

    GET  /api/files?path=sub/dir               listing
    GET  /api/files?path=sub/a.jpg&raw=1       the bytes, inline (preview)
    GET  /api/files?path=sub/a.jpg&download=1  the bytes, as an attachment
    GET  /api/files?path=sub/dir&zip=1         the folder, streamed as a zip
    POST /api/files/upload?path=sub/dir        multipart ``file`` (one or many)
    POST /api/files/mkdir                      json {"path": "sub/new"}
    POST /api/files/delete                     json {"path": "sub/a.jpg"}

Delete removes a file or an EMPTY folder only — emptying a folder of
captures is a deliberate sequence, never one mis-click. The server runs
under sudo, so every file and folder it creates is handed back to the
invoking user, or the operator could not delete it later.
"""
import json
import mimetypes
import os
import pwd
import zipfile
from pathlib import Path

import tornado.web


def default_captures_dir() -> Path:
    """``~/captures`` of the human who started the server — under sudo
    that is SUDO_USER's home, not /root."""
    user = os.environ.get("SUDO_USER")
    if user:
        try:
            return Path(pwd.getpwnam(user).pw_dir) / "captures"
        except KeyError:
            pass
    return Path.home() / "captures"


def hand_back(path: Path) -> None:
    """Give a file or folder the server created back to the invoking
    user. Never raises."""
    try:
        uid = int(os.environ.get("SUDO_UID", -1))
        gid = int(os.environ.get("SUDO_GID", -1))
        if uid >= 0:
            os.chown(path, uid, gid)
    except Exception:
        pass


def ensure_root(root: Path) -> Path:
    root = Path(root).expanduser().resolve()
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        hand_back(root)
    return root


def safe_join(base: Path, rel: str) -> Path:
    """``base/rel`` resolved, refusing anything that escapes ``base``."""
    target = (base / (rel or "").lstrip("/")).resolve()
    if target != base and base not in target.parents:
        raise ValueError("path is outside the captures folder")
    return target


def _entry(p: Path, base: Path) -> dict:
    st = p.stat()
    return {
        "name": p.name,
        "path": str(p.relative_to(base)),
        "dir": p.is_dir(),
        "size": 0 if p.is_dir() else st.st_size,
        "mtime": st.st_mtime,
    }


class _ZipSink:
    """Write-only, non-seekable file object for ``zipfile``: bytes pile
    up here and the handler drains them to the socket after each member,
    so a folder of thousands of captures streams instead of being built
    in memory. zipfile sees no ``seek`` and writes data descriptors."""
    def __init__(self):
        self.buf = bytearray()
        self.pos = 0

    def write(self, b):
        self.buf += b
        self.pos += len(b)
        return len(b)

    def tell(self):
        return self.pos

    def flush(self):
        pass

    def drain(self) -> bytes:
        out = bytes(self.buf)
        self.buf.clear()
        return out


async def stream_zip(handler: tornado.web.RequestHandler, folder: Path, name: str) -> None:
    """Stream ``folder`` as ``name.zip``. Stored, not deflated: captures
    are JPEG/PNG already, and compressing them again costs the Pi CPU
    for nothing."""
    handler.set_header("Content-Type", "application/zip")
    handler.set_header("Content-Disposition", f'attachment; filename="{name}.zip"')
    sink = _ZipSink()
    with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for fn in sorted(filenames):
                if fn.startswith("."):
                    continue
                p = Path(dirpath) / fn
                zf.write(p, arcname=str(Path(name) / p.relative_to(folder)))
                handler.write(sink.drain())
                await handler.flush()
    handler.write(sink.drain())      # the central directory
    await handler.flush()


class FilesHandler(tornado.web.RequestHandler):
    def initialize(self, root: Path):
        self.root = root

    def set_default_headers(self):
        self.set_header("Cache-Control", "no-store")

    async def get(self):
        try:
            rel = self.get_argument("path", "")
            target = safe_join(self.root, rel)

            if target.is_dir() and self.get_argument("zip", ""):
                name = target.name if target != self.root else self.root.name
                await stream_zip(self, target, name)
                return

            if target.is_file():
                if self.get_argument("raw", ""):
                    ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                    self.set_header("Content-Type", ctype)
                else:
                    self.set_header("Content-Type", "application/octet-stream")
                    self.set_header("Content-Disposition",
                                    f'attachment; filename="{target.name}"')
                with open(target, "rb") as fp:
                    while chunk := fp.read(1 << 16):
                        self.write(chunk)
                await self.flush()
                return

            # An absent folder lists EMPTY rather than erroring: "nothing
            # captured yet" is the honest answer.
            entries = [] if not target.exists() else sorted(
                (_entry(c, self.root) for c in target.iterdir() if not c.name.startswith(".")),
                key=lambda e: (not e["dir"], -e["mtime"]),
            )
            self.write({"abs": str(self.root), "path": rel, "entries": entries})
        except Exception as e:
            self.set_status(400)
            self.write({"error": str(e)})


class FilesActionHandler(tornado.web.RequestHandler):
    def initialize(self, root: Path):
        self.root = root

    def post(self, action):
        try:
            if action == "upload":
                files = (self.request.files or {}).get("file") or []
                if not files:
                    raise ValueError("No file uploaded")
                folder = safe_join(self.root, self.get_argument("path", ""))
                if not folder.exists():
                    folder.mkdir(parents=True, exist_ok=True)
                    hand_back(folder)
                saved = []
                for up in files:
                    filename = os.path.basename(up["filename"] or "")
                    if not filename:
                        continue
                    dest = safe_join(folder, filename)
                    with open(dest, "wb") as fp:
                        fp.write(up["body"])
                    hand_back(dest)
                    saved.append(dest.name)
                if not saved:
                    raise ValueError("Empty filename")
                self.write({"ok": True, "saved": saved})
                return

            body = json.loads(self.request.body or b"{}")
            rel = body.get("path") or ""
            target = safe_join(self.root, rel)

            if action == "mkdir":
                if not rel:
                    raise ValueError("name is required")
                target.mkdir(parents=True, exist_ok=True)
                hand_back(target)
                self.write({"ok": True, "path": str(target.relative_to(self.root))})
                return

            if action == "delete":
                if target == self.root:
                    raise ValueError("cannot delete the captures folder itself")
                if not target.exists():
                    raise ValueError("no such file")
                if target.is_dir():
                    if any(target.iterdir()):
                        raise ValueError("folder is not empty — empty it first")
                    target.rmdir()
                else:
                    target.unlink()
                self.write({"ok": True})
                return

            raise ValueError(f"unknown action: {action}")
        except Exception as e:
            self.set_status(400)
            self.write({"error": str(e)})
