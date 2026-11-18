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
captures is a deliberate sequence, never one mis-click. Every file and
folder created here is owned like the folder it is in
(dorna_vision/ownership.py): the operator's — the upgrade makes
``/home/dorna/captures`` dorna's and starts the service with it; a server
started by hand under sudo makes ``~/captures`` in the invoking user's
home.
"""
import mimetypes
import os
import pwd
from pathlib import Path

import tornado.web

from dorna_vision import ownership


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


def ensure_root(root: Path) -> Path:
    """The captures folder, made when missing, owned like the folder it is
    made in (the upgrade makes the service's beforehand)."""
    return Path(ownership.makedirs(Path(root).expanduser().resolve()))


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
        ownership.own(path)


class FilesSocket(fslive.FilesSocket):
    """``WS /ws/files`` — the captures folder, live (fslive.FilesSocket)."""

    def initialize(self, root: Path):
        self.root = root

    def resolve(self, root: str) -> Path:
        if root != ROOT_NAME:
            raise ValueError(f"unknown folder: {root}")
        return self.root

    def created(self, path: Path) -> None:
        ownership.own(path)
