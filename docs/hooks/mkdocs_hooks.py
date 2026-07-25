"""MkDocs build hooks for GrafitoDB.

Injects the package version into the docs build so the version badge shown in
the header stays in sync with every release automatically, with no manual step.

Single source of truth = ``pyproject.toml`` (via the installed package
metadata, with a filesystem fallback for local/uninstalled builds).
"""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version as _dist_version
from pathlib import Path

_PACKAGE_NAME = "grafitodb"


def _read_version() -> str:
    """Resolve the current package version.

    Prefers installed distribution metadata (this is how CI builds run, since
    the docs job installs the package with ``pip install -e ".[docs]"``); falls
    back to parsing ``pyproject.toml`` for editable/local builds where the
    metadata may be stale.
    """
    try:
        return _dist_version(_PACKAGE_NAME)
    except PackageNotFoundError:
        pass

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        return str(data["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return ""


def on_config(config):
    """Expose the version to templates as ``config.extra.grafito_version``.

    A dedicated key is used (not the theme-reserved ``extra.version``) so that
    Material's mike/version-selector machinery is never triggered.
    """
    grafito_version = _read_version()
    if grafito_version:
        extra = dict(config.get("extra") or {})
        extra["grafito_version"] = grafito_version
        config["extra"] = extra
    return config
