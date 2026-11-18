"""This unit's system — what ``upgrade`` installed, and running it — for a
vision Pi's System page (web/static/js/system.js).

The vision server shows a System page only where the unit declares it the
owner: ``/etc/dorna/unit.json`` ``{"system": "vision"}``, written by the
vision Pi's upgrade (branch ``vision_pro``). A Jetson declares
``"workspace"``: there the workspace's System page upgrades everything,
this server included, and this page is not offered. Declared, never
guessed.

The rules are the workspace's (``gui/orchestrator/system.py`` and
``project_git.part_state`` in the workspace repo), so both System pages
agree on what "to update" means:

NOTHING HERE KNOWS THE PARTS. The ``upgrade`` repo's top-level setup.sh
writes ``RECORD`` at the end of every upgrade: each part's entry is written
by the part itself (``lib/record.sh`` — name, label, folder, repo, branch,
commit, exit code). A part added to the upgrade appears after the next
upgrade.

THE UPGRADE IS ITS OWN SYSTEM JOB. It restarts this server and reboots the
unit, so it never runs as a child of this process — that would end it
halfway. ``systemd-run`` gives it its own unit (``UNIT``): it outlives this
server's restart, a second one cannot start while it runs, and its output
is in the journal, which outlives the reboot — the page reads it back.

Git runs as each repository's OWNER, never as root: the server is root,
the repos belong to the user who cloned them.
"""

from __future__ import annotations

import json
import os
import pwd
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

UNIT_DECLARATION = "/etc/dorna/unit.json"
RECORD = "/var/lib/dorna/upgrade.json"
HELPER = "/usr/local/bin/upgrade"
UNIT = "dorna-upgrade"
FETCH_TIMEOUT_S = 30
GIT_TIMEOUT_S = 60


def unit() -> dict:
    """The unit's declaration, ``{}`` before the upgrade has written it."""
    try:
        with open(UNIT_DECLARATION) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def owned() -> bool:
    """Does this unit give the System page to the vision server?"""
    return unit().get("system") == "vision"


# ── a part's folder vs what the upgrade installs (as the workspace's
#    project_git.part_state) ──────────────────────────────────────────────

class GitError(RuntimeError):
    """A git command failed — its stderr is the message."""


def _git(cwd: str, *args: str, timeout: float = GIT_TIMEOUT_S, check: bool = True):
    """``git -C cwd args…`` as the folder's owner. Returns (rc, stdout, stderr)."""
    env = ["GIT_TERMINAL_PROMPT=0",
           "GIT_SSH_COMMAND=ssh -o BatchMode=yes -o ConnectTimeout=10"]
    owner = pwd.getpwuid(os.stat(cwd).st_uid).pw_name
    if os.geteuid() == 0 and owner != "root":
        cmd = ["sudo", "-u", owner, "-H", "env", *env, "git", "-C", cwd, *args]
    else:
        cmd = ["env", *env, "git", "-C", cwd, *args]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise GitError(f"git {' '.join(args)}: no answer in {timeout:g} s")
    if check and p.returncode != 0:
        raise GitError((p.stderr or p.stdout).strip() or f"git {' '.join(args)} failed ({p.returncode})")
    # rstrip only: status --porcelain lines START with a status column.
    return p.returncode, p.stdout.rstrip(), p.stderr.strip()


def _commit(repo: str, ref: str) -> Optional[dict]:
    rc, out, _ = _git(repo, "log", "-1", "--format=%h%x1f%s%x1f%cs", ref, check=False)
    if rc != 0 or not out:
        return None
    h, subject, date = (out.split("\x1f") + ["", "", ""])[:3]
    return {"hash": h, "subject": subject, "date": date}


def part_state(path: str, branch: str, fetch: bool = True) -> dict:
    """How the folder ``path`` compares with ``origin/<branch>`` — what the
    part's script makes it (``fetch``, ``checkout -B``, ``reset --hard``,
    ``clean -fd``), so what that would cost: ``changes`` (uncommitted
    edits — discarded), ``untracked`` (deleted; ignored files are kept),
    ``switches`` (on another branch), ``lost_commits`` (local commits the
    remote does not have). ``behind`` counts the remote's commits this
    folder does not have; a failed fetch is ``reachable: False``."""
    if not os.path.isdir(path):
        return {"exists": False}
    rc, repo, err = _git(path, "rev-parse", "--show-toplevel", check=False)
    if rc != 0:
        return {"exists": True, "git": False, "error": err or "not a git repository"}
    _, current, _ = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    out = {"exists": True, "git": True, "branch": current, "head": _commit(repo, "HEAD"),
           "target_branch": branch, "reachable": None, "fetch_error": ""}
    if fetch and branch:
        rc, _, err = _git(repo, "fetch", "origin", branch, timeout=FETCH_TIMEOUT_S, check=False)
        out["reachable"] = rc == 0
        out["fetch_error"] = "" if rc == 0 else err
    target = f"origin/{branch}"
    rc, _, _ = _git(repo, "rev-parse", "--verify", "--quiet", target, check=False)
    out["target"] = _commit(repo, target) if (rc == 0 and branch) else None
    ahead = behind = 0
    lost: List[dict] = []
    if out["target"]:
        _, counts, _ = _git(repo, "rev-list", "--left-right", "--count", f"HEAD...{target}")
        ahead, behind = (int(x) for x in counts.split())
        if ahead and current == branch:
            _, log, _ = _git(repo, "log", "--format=%h%x1f%s", f"{target}..HEAD")
            lost = [dict(zip(("hash", "subject"), l.split("\x1f"))) for l in log.splitlines() if l]
    _, st, _ = _git(repo, "status", "--porcelain", "--untracked-files=all")
    lines = [l for l in st.splitlines() if l.strip()]
    out.update(behind=behind, ahead=ahead, switches=bool(branch) and current != branch,
               lost_commits=lost,
               changes=[l[3:] for l in lines if not l.startswith("??")],
               untracked=[l[3:] for l in lines if l.startswith("??")])
    return out


# ── the unit and its upgrade ─────────────────────────────────────────────

def _record() -> Optional[dict]:
    try:
        with open(RECORD) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:                      # present but unreadable: say so
        return {"error": f"{RECORD}: {e}"}


def running() -> bool:
    """Is an upgrade started from here still running?"""
    return subprocess.run(["systemctl", "is-active", "--quiet", UNIT]).returncode == 0


def _uptime_s() -> Optional[float]:
    try:
        with open("/proc/uptime") as f:
            return float(f.read().split()[0])
    except Exception:
        return None


def needs_update(part: dict) -> bool:
    """THE rule for "this part is not up to date" (the workspace's): its
    folder is missing, behind what the upgrade installs, or on another
    branch. Unknown (no folder, the remote not reachable) is not counted."""
    if not part.get("dir"):
        return False
    n = part.get("now") or {}
    if not n.get("exists", True):
        return True
    if not n.get("git") or n.get("reachable") is False:
        return False
    return bool(n.get("behind") or n.get("switches"))


def state(fetch: bool = True) -> dict:
    """The unit, the last upgrade's record, and for each part the record
    lists, how its folder compares with what the upgrade installs (fetched
    in parallel)."""
    rec = _record()
    parts = list((rec or {}).get("parts") or [])

    def one(p):
        e = dict(p)
        if p.get("dir"):
            try:
                e["now"] = part_state(p["dir"], p.get("branch") or "", fetch)
            except GitError as ex:
                e["now"] = {"exists": True, "git": False, "error": str(ex)}
        return e

    with ThreadPoolExecutor(max_workers=max(1, len(parts))) as pool:
        parts = list(pool.map(one, parts))
    return {
        "to_update": [p["name"] for p in parts if needs_update(p)],
        "uptime_s": _uptime_s(),
        "helper": os.path.isfile(HELPER) and os.access(HELPER, os.X_OK),
        "helper_path": HELPER,
        "record": {k: v for k, v in (rec or {}).items() if k != "parts"} if rec else None,
        "parts": parts,
        "running": running(),
        "unit": UNIT,
        "now": time.time(),
    }


def start() -> dict:
    """Start ``upgrade`` as its own system job. Refused when this unit does
    not give the System page to the vision server, when the helper is not
    installed, or when one is already running — everything else (connected
    clients, open cameras, local changes) is the page's warning, confirmed
    past."""
    if not owned():
        raise RuntimeError("this unit's upgrade is not the vision server's to run "
                           f"({UNIT_DECLARATION} does not declare \"system\": \"vision\")")
    if not (os.path.isfile(HELPER) and os.access(HELPER, os.X_OK)):
        raise RuntimeError(f"{HELPER} is not installed on this unit — run the first-time "
                           f"upgrade from a terminal (the upgrade repo's README)")
    if running():
        raise RuntimeError("an upgrade is already running")
    started = time.time()
    p = subprocess.run(
        ["systemd-run", f"--unit={UNIT}", "--collect",
         "--description=Dorna upgrade (vision System page)", HELPER],
        capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout).strip() or "systemd-run failed")
    return {"started_at": started, "unit": UNIT}


def log(since: float, cursor: str = "") -> dict:
    """The upgrade's output since ``since`` (epoch s) — or after ``cursor``
    — from the journal: it survives this server's restart and the reboot."""
    cmd = ["journalctl", "-u", UNIT, "-o", "json", "--no-pager"]
    cmd += ["--after-cursor", cursor] if cursor else ["--since", f"@{int(since)}"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    lines, last = [], cursor
    for raw in p.stdout.splitlines():
        try:
            e = json.loads(raw)
        except ValueError:
            continue
        msg = e.get("MESSAGE", "")
        if isinstance(msg, list):               # non-UTF-8 output arrives as bytes
            msg = bytes(msg).decode("utf-8", "replace")
        lines.append({"t": int(e.get("__REALTIME_TIMESTAMP", "0")) / 1e6, "text": msg})
        last = e.get("__CURSOR", last)
    return {"lines": lines, "cursor": last, "running": running()}
