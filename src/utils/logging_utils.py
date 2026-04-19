"""Advanced logging utilities for the stress detection project."""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar


F = TypeVar("F", bound=Callable[..., Any])


class StructuredFormatter(logging.Formatter):
    """Format log records as JSON structured logs."""

    def __init__(self, timestamp_format: str = "%Y-%m-%dT%H:%M:%S.%fZ") -> None:
        super().__init__()
        self.timestamp_format = timestamp_format

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record into JSON."""

        payload: Dict[str, Any] = {
            "timestamp": self._format_timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "context") and isinstance(record.context, dict):
            payload["context"] = record.context

        return json.dumps(payload, ensure_ascii=True, default=str)

    def _format_timestamp(self, created: float) -> str:
        """Format UNIX timestamp in UTC."""

        dt = datetime.fromtimestamp(created, tz=timezone.utc)
        return dt.strftime(self.timestamp_format)


@dataclass
class LoggerConfig:
    """Configuration container for the logger manager."""

    logger_name: str = "stress_detection"
    level: int = logging.INFO
    log_file: Path = Path("logs/system.log")
    console_enabled: bool = True
    file_enabled: bool = True
    structured_logs: bool = False
    propagate: bool = False
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 5
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"
    message_format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


class LoggerManager:
    """Manage application-wide logging with console, file, and performance logging support."""

    def __init__(self, config: Optional[LoggerConfig] = None) -> None:
        """Initialize the logger manager."""

        self.config = config or LoggerConfig()
        self._logger = logging.getLogger(self.config.logger_name)
        self._logger.setLevel(self.config.level)
        self._logger.propagate = self.config.propagate
        self._configure_handlers()

    def get_logger(self) -> logging.Logger:
        """Return the configured logger instance."""

        return self._logger

    def log_structured(
        self,
        level: int,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        exc_info: Any = None,
    ) -> None:
        """Log a message with optional structured context."""

        extra = {"context": context or {}}
        self._logger.log(level, message, extra=extra, exc_info=exc_info)

    def log_error(
        self,
        message: str,
        *,
        error: Optional[BaseException] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an error message with exception context."""

        merged_context = dict(context or {})
        if error is not None:
            merged_context["error_type"] = error.__class__.__name__
            merged_context["error_message"] = str(error)

        self.log_structured(
            logging.ERROR,
            message,
            context=merged_context,
            exc_info=error if error is not None else None,
        )

    def log_performance(
        self,
        operation: str,
        duration_seconds: float,
        *,
        level: int = logging.INFO,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log performance timing information."""

        merged_context = dict(context or {})
        merged_context.update(
            {
                "operation": operation,
                "duration_seconds": round(duration_seconds, 6),
                "duration_ms": round(duration_seconds * 1000.0, 3),
            }
        )
        self.log_structured(level, f"Performance log for '{operation}'", context=merged_context)

    def execution_time(self, *, level: int = logging.INFO, include_args: bool = False) -> Callable[[F], F]:
        """Create a decorator that logs execution time for a callable."""

        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                context: Dict[str, Any] = {"function": func.__qualname__}

                if include_args:
                    context["args_count"] = len(args)
                    context["kwargs_keys"] = sorted(kwargs.keys())

                try:
                    result = func(*args, **kwargs)
                except Exception as exc:
                    duration = time.perf_counter() - start
                    context["status"] = "failed"
                    self.log_performance(func.__qualname__, duration, level=logging.ERROR, context=context)
                    self.log_error(
                        f"Execution failed for '{func.__qualname__}'",
                        error=exc,
                        context=context,
                    )
                    raise

                duration = time.perf_counter() - start
                context["status"] = "success"
                self.log_performance(func.__qualname__, duration, level=level, context=context)
                return result

            return wrapper  # type: ignore[return-value]

        return decorator

    def set_level(self, level: int) -> None:
        """Update logger and handler levels."""

        self.config.level = level
        self._logger.setLevel(level)
        for handler in self._logger.handlers:
            handler.setLevel(level)

    def _configure_handlers(self) -> None:
        """Configure console and file handlers without duplication."""

        self._logger.handlers.clear()

        formatter = self._build_formatter()

        if self.config.console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.config.level)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

        if self.config.file_enabled:
            self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                filename=self.config.log_file,
                maxBytes=self.config.max_bytes,
                backupCount=self.config.backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(self.config.level)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def _build_formatter(self) -> logging.Formatter:
        """Build the appropriate formatter based on configuration."""

        if self.config.structured_logs:
            return StructuredFormatter(timestamp_format="%Y-%m-%dT%H:%M:%S.%fZ")

        return logging.Formatter(
            fmt=self.config.message_format,
            datefmt=self.config.timestamp_format,
        )


def execution_time(logger_manager: LoggerManager, *, level: int = logging.INFO, include_args: bool = False) -> Callable[[F], F]:
    """Module-level decorator helper for logging execution time."""

    return logger_manager.execution_time(level=level, include_args=include_args)


__all__ = [
    "LoggerConfig",
    "LoggerManager",
    "StructuredFormatter",
    "execution_time",
]
