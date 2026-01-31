"""Utilities for reproducible experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
import platform
import random
from typing import Any, Dict, Iterable, Optional
import uuid

import numpy as np


@dataclass(frozen=True)
class RunContext:
    run_id: str
    seed: Optional[int]


class _RunContextFilter(logging.Filter):
    def __init__(self, context: RunContext) -> None:
        super().__init__()
        self._context = context

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.run_id = self._context.run_id
        record.seed = (
            self._context.seed if self._context.seed is not None else "unknown"
        )
        return True


def generate_run_id(prefix: str | None = None, now: datetime | None = None) -> str:
    """Return a unique run identifier with an optional prefix."""
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    token = uuid.uuid4().hex[:8]
    if prefix:
        return f"{prefix}-{timestamp}-{token}"
    return f"{timestamp}-{token}"


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def create_run_metadata(
    *,
    run_id: str,
    seed: Optional[int],
    now: datetime | None = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect run metadata for logging or persistence."""
    timestamp = (now or datetime.now(timezone.utc)).isoformat()
    metadata: Dict[str, Any] = {
        "run_id": run_id,
        "seed": seed,
        "started_at": timestamp,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
    if extra:
        metadata.update(extra)
    return metadata


def configure_logging(
    *,
    run_id: str,
    seed: Optional[int],
    level: int | str = logging.INFO,
    logger_name: str = "shap-it-like-its-hot",
    log_file: str | os.PathLike[str] | None = None,
    stream: Optional[Any] = None,
    force: bool = False,
) -> logging.Logger:
    """Configure a logger with a consistent run context format."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if force:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    if logger.handlers:
        return logger

    context = RunContext(run_id=run_id, seed=seed)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s run_id=%(run_id)s seed=%(seed)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers: Iterable[logging.Handler]
    stream_handler = logging.StreamHandler(stream=stream)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(_RunContextFilter(context))
    handlers = [stream_handler]

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_RunContextFilter(context))
        handlers = [stream_handler, file_handler]

    for handler in handlers:
        logger.addHandler(handler)

    return logger
