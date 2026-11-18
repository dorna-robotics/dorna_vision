"""
Who owns what this package writes — ONE rule: a file or folder it creates
takes the owner of the folder it is created in.

The vision server runs as root (a systemd service; by hand, under sudo),
and so does a robot program in Jupyter. Without this every capture, saved
image and log would be root's, and the operator could not open or delete
it from a shell or Jupyter. The folder already says who it belongs to: the
operator's captures folder (/home/dorna/captures on a unit), a project
folder, a home. Nothing is read from the environment (SUDO_UID is unset in
a service), nothing is guessed.

    makedirs(path)  the folder and any missing parents, each owned like the
                    folder it was made in; returns path
    own(path)       the file or folder: its parent folder's owner

Not root, there is nothing to change (a process owns what it creates).
Never raises: ownership never fails a capture.

The client package (dorna_vision_client) keeps a copy of this rule — it is
installed on the workspace computer without this package.
"""

import os


def own(path):
    """Give ``path`` the owner of the folder it is in. Never raises."""
    if os.geteuid() != 0:
        return
    try:
        st = os.stat(os.path.dirname(os.path.abspath(path)))
        os.chown(path, st.st_uid, st.st_gid)
    except Exception:
        pass


def makedirs(path):
    """``os.makedirs(path, exist_ok=True)``, each folder it creates owned like
    the folder it was made in. Returns ``path``."""
    path = os.path.abspath(path)
    missing = []
    p = path
    while not os.path.isdir(p):
        missing.append(p)
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    os.makedirs(path, exist_ok=True)
    for p in reversed(missing):      # top down: each takes its parent's owner, already set
        own(p)
    return path
