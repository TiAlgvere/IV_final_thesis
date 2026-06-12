"""Small shared utilities (results directories, JSON dumping)."""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any


def results_dir(phase: str, root: str = "results") -> str:
    """Create and return a timestamped results subfolder results/<phase>_<ts>/."""
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(root, f"{phase}_{ts}")
    os.makedirs(path, exist_ok=True)
    return path


def dump_json(obj: Any, path: str) -> None:
    """Dump a dataclass / dict / list to pretty JSON (dataclasses supported)."""
    def _default(o: Any):
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        if isinstance(o, complex):
            return {"real": o.real, "imag": o.imag}
        return str(o)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_default)
