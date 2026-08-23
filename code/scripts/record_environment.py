"""Print a compact JSON record of the active reproduction environment."""

from __future__ import annotations

import importlib
import json
import platform
import shutil
import sys


def version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    return getattr(module, "__version__", "unknown")


record = {
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "packages": {
        name: version(name)
        for name in ("numpy", "scipy", "matplotlib", "wfdb", "mne", "xlrd", "requests")
    },
    "commands": {
        name: shutil.which(name)
        for name in ("make", "latexmk", "pdflatex", "bibtex", "pdfinfo")
    },
}
print(json.dumps(record, indent=2))

