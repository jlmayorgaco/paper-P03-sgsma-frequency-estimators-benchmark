from __future__ import annotations

import sys
from pathlib import Path


def _prioritize_local_src() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    src_str = str(src_path)
    if src_str in sys.path:
        sys.path.remove(src_str)
    sys.path.insert(0, src_str)

    # Avoid cross-repo import pollution when modules were preloaded from
    # another checkout that also exposes `estimators`/`scenarios`.
    for mod_name in list(sys.modules):
        if mod_name == "estimators" or mod_name.startswith("estimators."):
            sys.modules.pop(mod_name, None)
        if mod_name == "scenarios" or mod_name.startswith("scenarios."):
            sys.modules.pop(mod_name, None)


_prioritize_local_src()
