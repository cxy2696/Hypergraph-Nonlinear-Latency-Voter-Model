"""Run provenance: timestamps, versions, git state, checksums.

Every artifact written by ``python -m hnl.cli reproduce`` is accompanied by a
manifest so that a reported number can be traced back to the code, the data
and the parameters that produced it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: Iterable[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_info(root: str | Path) -> Dict[str, Any]:
    root = Path(root)
    commit = _run(["git", "rev-parse", "HEAD"], cwd=root)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    status = _run(["git", "status", "--porcelain"], cwd=root)
    # Generated artifacts (results/, figures/) are expected to change when the
    # pipeline runs; they must not mark the *source* tree as dirty.
    if status is not None:
        relevant = [
            line
            for line in status.splitlines()
            if not line[3:].startswith(("results/", "figures/"))
        ]
        status = "\n".join(relevant)
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


def environment_info() -> Dict[str, Any]:
    import matplotlib
    import numpy
    import pandas

    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "matplotlib": matplotlib.__version__,
    }


def build_manifest(
    root: str | Path,
    argv: Iterable[str],
    designs: Dict[str, Any],
    datasets: Dict[str, Any],
    artifacts: Iterable[str | Path],
) -> Dict[str, Any]:
    root = Path(root)
    files = []
    for artifact in artifacts:
        path = Path(artifact)
        if not path.exists() or path.is_dir():
            continue
        files.append(
            {
                "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "created_utc": utc_now(),
        "argv": list(argv),
        "git": git_info(root),
        "environment": environment_info(),
        "designs": designs,
        "datasets": datasets,
        "artifacts": files,
    }


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path
