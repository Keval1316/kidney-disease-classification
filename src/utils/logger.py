"""
src/utils/logger.py
--------------------
Project-wide logging setup.

Features
--------
- Consistent format: timestamp | level | module | message
- Logs to stdout (always)
- Logs to a rotating file under logs/kidney_clf.log (auto-created)
  -> 5 MB max per file, keeps 3 backup files
- Propagation disabled so we never double-log through the root logger
- get_logger() is idempotent -- safe to call multiple times for the same name

Usage
-----
    from src.utils.logger import get_logger
    logger = get_logger(__name__)

    logger.info("Pipeline started")
    logger.warning("Class imbalance detected")
    logger.error("File not found: %s", path)
    logger.exception("Unexpected error")   # prints full traceback
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "kidney_clf.log"
LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 5 * 1024 * 1024   # 5 MB
BACKUP_COUNT = 3


def _ensure_log_dir() -> None:
    """Create the logs/ directory if it does not exist."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger configured with:
      - StreamHandler  -> stdout
      - RotatingFileHandler -> logs/kidney_clf.log

    Calling this multiple times with the same name is safe (handlers are not
    duplicated).

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.
    level : int
        Logging level (default INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Already configured -- return as-is to avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False   # prevent double output through root logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # -----------------------------------------------------------------------
    # Handler 1 -- stdout
    # -----------------------------------------------------------------------
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(level)
    logger.addHandler(stdout_handler)

    # -----------------------------------------------------------------------
    # Handler 2 -- rotating file (best-effort; skip silently if dir unwritable)
    # -----------------------------------------------------------------------
    try:
        _ensure_log_dir()
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    except OSError:
        # Running in a read-only environment (e.g., some CI containers) --
        # stdout-only logging is still fully functional.
        pass

    return logger


# ---------------------------------------------------------------------------
# Convenience log helpers
# ---------------------------------------------------------------------------

def log_section(logger: logging.Logger, title: str) -> None:
    """Print a clearly visible section header in the log."""
    border = "=" * 60
    logger.info(border)
    logger.info("  %s", title)
    logger.info(border)


def log_dict(logger: logging.Logger, data: dict, title: str = "") -> None:
    """Log every key/value pair in *data* at INFO level."""
    if title:
        logger.info(title)
    for k, v in data.items():
        logger.info("  %-30s = %s", k, v)