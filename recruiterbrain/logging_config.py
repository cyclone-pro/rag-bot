"""Centralized logging configuration for recruiter brain."""
from __future__ import annotations

import logging
import os
from typing import Optional

from recruiterbrain.env_loader import load_env

DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logging once using env overrides."""
    load_env()
    if logging.getLogger().handlers:
        return

    log_level = level or os.environ.get("RB_LOG_LEVEL", "INFO").upper()
    log_format = os.environ.get("RB_LOG_FORMAT", DEFAULT_FORMAT)

    logging.basicConfig(level=log_level, format=log_format)
