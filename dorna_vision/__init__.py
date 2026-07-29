from .ai import *
from .board import *
from .calibration import *
from .detect import *
from .draw import *
from .find import *
from .pose import *
from .util import *
from .visual import *
from .conversion import *
from . import grasp

def _version(base="2.4"):
    """Auto-version from git: base.<commit count>+<short sha>.

    The patch number is the commit count on the checked-out branch, so
    the version changes on EVERY commit with no manual bump; the sha
    pins the exact build. Deployed units are git clones (the upgrade
    syncs by fetch/reset), so this is always available there; a
    repo-less copy falls back to the bare base.
    """
    import os
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        n = subprocess.check_output(
            ["git", "-C", root, "rev-list", "--count", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
        sha = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
        return f"{base}.{n}+{sha}"
    except Exception:
        return base


__version__ = _version()