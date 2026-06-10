"""Application logging setup: daily files, timestamps, crash handlers."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

FORMATS = {
    "simple": "%(asctime)s - %(levelname)s - %(message)s",
    "detailed": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    "timestamp": "%(asctime)s - %(levelname)s - %(message)s",
    "minimal": "%(levelname)s: %(message)s",
}

LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


def _resolve_log_file_path(project_root: Path, log_to_file: bool) -> str | None:
    if not log_to_file:
        return None

    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    default_log_file = str(logs_dir / "app.log")
    return os.getenv("LOG_FILE_PATH", default_log_file)


def _build_handlers(log_file_path: str | None, formatter: logging.Formatter) -> list[logging.Handler]:
    handlers: list[logging.Handler] = []

    if log_file_path:
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        retention_days = int(os.getenv("LOG_RETENTION_DAYS", "30"))
        file_handler = TimedRotatingFileHandler(
            filename=log_file_path,
            when="midnight",
            interval=1,
            backupCount=retention_days,
            encoding="utf-8",
        )
        file_handler.suffix = "%Y-%m-%d"
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    return handlers


def _configure_uvicorn_loggers(handlers: list[logging.Handler], level: int) -> None:
    for name in UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = False
        for handler in handlers:
            uvicorn_logger.addHandler(handler)
        uvicorn_logger.setLevel(level)


def setup_logging(project_root: Path | None = None) -> logging.Logger:
    """Configure root logging from environment variables."""
    project_root = project_root or Path(__file__).resolve().parent.parent

    log_level_name = os.getenv("LOG_LEVEL", "ERROR").upper()
    log_format_name = os.getenv("LOG_FORMAT", "timestamp")
    log_to_file = os.getenv("LOG_TO_FILE", "true").lower() == "true"

    log_level = LEVEL_MAP.get(log_level_name, logging.ERROR)
    log_format_string = FORMATS.get(log_format_name, FORMATS["timestamp"])
    formatter = logging.Formatter(log_format_string, datefmt=DATE_FORMAT)

    log_file_path = _resolve_log_file_path(project_root, log_to_file)
    handlers = _build_handlers(log_file_path, formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(log_level)
    for handler in handlers:
        root_logger.addHandler(handler)

    _configure_uvicorn_loggers(handlers, log_level)
    install_crash_handlers()

    app_logger = logging.getLogger("main")
    if log_file_path:
        app_logger.info(
            "Logging configured: level=%s format=%s file=%s",
            log_level_name,
            log_format_name,
            log_file_path,
        )
    else:
        app_logger.info(
            "Logging configured: level=%s format=%s (console only)",
            log_level_name,
            log_format_name,
        )

    return app_logger


def install_crash_handlers() -> None:
    """Log uncaught main-thread exceptions before process exit."""
    crash_logger = logging.getLogger("crash")

    def _uncaught(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        crash_logger.critical(
            "Uncaught exception — process exiting",
            exc_info=(exc_type, exc_value, exc_tb),
        )
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _uncaught

    def _signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        crash_logger.info("Received signal %s (%s)", sig_name, signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            # Not available on all platforms or when not in main thread.
            pass


def install_asyncio_exception_handler() -> None:
    """Log unhandled asyncio task exceptions."""
    loop = asyncio.get_running_loop()
    crash_logger = logging.getLogger("crash")

    def _handler(_loop, context):
        message = context.get("message", "asyncio exception")
        exc = context.get("exception")
        if exc is not None:
            crash_logger.error("Asyncio error: %s", message, exc_info=exc)
        else:
            crash_logger.error("Asyncio error: %s", message)

    loop.set_exception_handler(_handler)
