from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent


def project_root() -> Path:
    """Return the checkout root when running from source, otherwise the caller cwd."""
    if SOURCE_ROOT.name == "src" and (SOURCE_ROOT.parent / "pyproject.toml").exists():
        return SOURCE_ROOT.parent
    return Path.cwd().resolve()


PROJECT_ROOT = project_root()
