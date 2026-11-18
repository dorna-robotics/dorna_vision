"""files.py — the Files page backend: the captures folder, live.

One root, the captures folder (``~/captures`` of the user who started
the server, ``--captures`` overrides). The work is ``fslive.py`` (twin of
the workspace orchestrator's); this file only names the root:

    WS   /ws/files                             open / mkdir / delete, and the
                                               changes pushed as they happen
                                               (a new capture appears at once)
    GET  /api/files?path=sub/a.jpg&raw=1       the bytes, inline (preview; cached
                                               when the URL carries its mtime)
    GET  /api/files?path=sub/a.jpg&download=1  the bytes, as an attachment
    GET  /api/files?path=sub/dir&zip=1         the folder, streamed as a zip
    PUT  /api/files/upload?path=sub&name=a.jpg the file as the body, streamed to disk

Delete removes a file or an EMPTY folder only — emptying a folder of
captures is a deliberate sequence, never one mis-click. The server runs
as root (a systemd service, or by hand under sudo), so every file and
folder it creates here is given the captures folder's own owner — the
operator's: the upgrade makes ``/home/dorna/captures`` dorna's and starts
the service with it; a server started by hand under sudo makes its
``~/captures`` the invoking user's — or the operator could not touch it
from Jupyter or a shell.
"""
import mimetypes
import os
import pwd
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


def hand_back(path: Path, root: Path) -> None:
    """Give a file or folder the server created inside the captures folder
    ``root`` that folder's owner. Never raises."""
    try:
        st = root.stat()
        os.chown(path, st.st_uid, st.st_gid)
    except Exception:
        pass


def ensure_root(root: Path) -> Path:
    """The captures folder, made when missing — by a server started by hand
    under sudo, for the invoking user (SUDO_UID); the service's is made by
    the upgrade, the operator's already."""
    root = Path(root).expanduser().resolve()
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        try:
            uid = int(os.environ.get("SUDO_UID", -1))
            if uid >= 0:
                os.chown(root, uid, int(os.environ.get("SUDO_GID", -1)))
        except Exception:
            pass
    return root


from . import fslive                          # noqa: E402

ROOT_NAME = "captures"      # the socket's one root


class FilesHandler(tornado.web.RequestHandler):
    """The BYTES of the captures (listing and actions are the socket's)."""

    def initialize(self, root: Path):
        self.root = root

    async def get(self):
        try:
            target = fslive.safe_join(self.root, self.get_argument("path", ""))
            if target.is_dir() and self.get_argument("zip", ""):
                name = target.name if target != self.root else self.root.name
                self.set_header("Cache-Control", "no-store")
                await fslive.send_zip(self, target, name)
                return
            if not target.is_file():
                raise ValueError("no such file")
            if self.get_argument("raw", ""):
                # The page puts the file's mtime in the URL (``t=``): the
                # bytes behind such a URL never change, so the browser may
                # keep them — stepping back through captures costs nothing.
                self.set_header("Cache-Control", "private, max-age=86400" if self.get_argument("t", "")
                                else "no-store")
                await fslive.send_file(self, target, inline=True,
                                       ctype=mimetypes.guess_type(target.name)[0])
                return
            self.set_header("Cache-Control", "no-store")
            await fslive.send_file(self, target)
        except Exception as e:
            if not self._headers_written:
                self.set_status(400)
                self.write({"error": str(e)})


class UploadHandler(fslive.UploadHandler):
    """``PUT /api/files/upload?path=<folder>&name=<file>`` (fslive)."""

    def initialize(self, root: Path):
        self.root = root

    def resolve(self, *args) -> Path:
        return self.root

    def created(self, path: Path) -> None:
        hand_back(path, self.root)


class FilesSocket(fslive.FilesSocket):
    """``WS /ws/files`` — the captures folder, live (fslive.FilesSocket)."""

    def initialize(self, root: Path):
        self.root = root

    def resolve(self, root: str) -> Path:
        if root != ROOT_NAME:
            raise ValueError(f"unknown folder: {root}")
        return self.root

    def created(self, path: Path) -> None:
        hand_back(path, self.root)
