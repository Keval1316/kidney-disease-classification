"""
src/utils/common.py
--------------------
Shared utility functions used across the pipeline.

These are lightweight helpers that do NOT import TensorFlow or MLflow so that
they can be safely imported from any module (including tests and the API).
"""

import json
import time
import functools
from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    logger.debug("Loaded YAML: %s", path)
    return data or {}


def save_yaml(data: dict, path: str | Path) -> None:
    """Write *data* to a YAML file, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    logger.debug("Saved YAML: %s", path)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Write *data* as JSON, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    logger.debug("Saved JSON: %s", path)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and parents) if it does not exist. Return a Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_size_mb(path: str | Path) -> float:
    """Return the size of a file in MB (0.0 if the file doesn't exist)."""
    p = Path(path)
    if not p.exists():
        return 0.0
    return p.stat().st_size / (1024 * 1024)


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------

def timed(fn):
    """
    Decorator that logs how long a function takes to run.

    Usage::

        @timed
        def my_function():
            ...
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        elapsed = time.time() - start
        logger.info("%s completed in %.1f s", fn.__qualname__, elapsed)
        return result
    return wrapper


# ---------------------------------------------------------------------------
# Class name utilities
# ---------------------------------------------------------------------------

def get_class_names(class_index: dict[str, int]) -> list[str]:
    """
    Convert a {class_name: index} mapping to an ordered list of class names.

    Parameters
    ----------
    class_index : dict
        e.g. {"Cyst": 0, "Normal": 1, "Stone": 2, "Tumor": 3}

    Returns
    -------
    list[str]
        e.g. ["Cyst", "Normal", "Stone", "Tumor"]
    """
    return [name for name, _ in sorted(class_index.items(), key=lambda x: x[1])]
