"""
logging_utils.py — Structured logging configuration.

We use structured JSON logging in production for:
- Searchability in log aggregation tools (Datadog, CloudWatch)
- Machine-parseable log entries
- Consistent format across services
"""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure root logger with appropriate format and level."""
    
    if json_format:
        # In production, use JSON for log aggregation
        try:
            import structlog
            structlog.configure(
                processors=[
                    structlog.processors.add_log_level,
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.JSONRenderer(),
                ],
                wrapper_class=structlog.BoundLogger,
                logger_factory=structlog.PrintLoggerFactory(),
            )
        except ImportError:
            pass  # Fallback to standard logging
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # Reduce noise from verbose libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
