from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src import config


def _formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_logging(name: str = "resumefit", level: int = logging.INFO, log_dir: Path | None = None) -> logging.Logger:
    log_dir = Path(log_dir) if log_dir else config.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    console = logging.StreamHandler()
    console.setFormatter(_formatter())
    logger.addHandler(console)

    file_handler = RotatingFileHandler(
        log_dir / f"{name}.log",
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(_formatter())
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    module = name or os.environ.get("RESUMEFIT_LOG_NAME", "resumefit")
    return logging.getLogger(module)


def configure_root() -> None:
    setup_logging("resumefit")
    for logger_name in ("hiring", "interviews", "calibration", "app", "auth"):
        child = logging.getLogger(f"resumefit.{logger_name}")
        if not child.handlers:
            child.setLevel(logging.INFO)
            child.addHandler(logging.getLogger("resumefit").handlers[0])
            child.propagate = False
