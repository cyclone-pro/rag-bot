"""Helper to load .env files so credentials are available to every module."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

from dotenv import load_dotenv

_LOADED_PATHS: List[Path] = []


def _candidate_paths(extra_paths: Optional[Iterable[str | os.PathLike[str]]] = None) -> List[Path]:
    """Build the ordered list of .env paths to try, with RB_ENV_PATH taking priority."""
    base_dir = Path(__file__).resolve().parent
    candidates: List[Path] = []

    if extra_paths:
        for path in extra_paths:
            if path:
                candidates.append(Path(path))

    rb_env_path = os.environ.get("RB_ENV_PATH")
    if rb_env_path:
        candidates.append(Path(rb_env_path))

    candidates.extend([base_dir.parent / ".env", base_dir / ".env"])

    seen = set()
    deduped: List[Path] = []
    for path in candidates:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def load_env(
    extra_paths: Optional[Iterable[str | os.PathLike[str]]] = None,
    *,
    force: bool = False,
) -> List[Path]:
    """
    Load environment variables from .env files (if present) into os.environ.
    Returns the list of files that were loaded.
    """
    global _LOADED_PATHS
    if _LOADED_PATHS and not force:
        return _LOADED_PATHS

    loaded: List[Path] = []
    for env_path in _candidate_paths(extra_paths):
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            loaded.append(env_path)

    _LOADED_PATHS = loaded
    return _LOADED_PATHS


def loaded_env_paths() -> List[Path]:
    """Return the list of .env files that have been loaded (if any)."""
    return list(_LOADED_PATHS)
