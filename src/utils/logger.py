"""
Logging utility for the Intelligent Attendance Monitoring System.

Provides a centralized logging configuration for the application.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Add project root to path for standalone execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

import config


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Set up and configure a logger for the application.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               If None, uses config.LOG_LEVEL
        log_to_file: Whether to log to file in addition to console
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set log level
    log_level = level or config.LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if log_to_file:
        try:
            config.ensure_directories()
            log_file = config.LOGS_DIR / "app.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not set up file logging: {e}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with default configuration.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
    
    Returns:
        Logger instance
    """
    return setup_logger(name)


# Application-wide logger
app_logger = setup_logger("attendance_system")


if __name__ == "__main__":
    # Test logging
    test_logger = get_logger(__name__)
    test_logger.debug("Debug message")
    test_logger.info("Info message")
    test_logger.warning("Warning message")
    test_logger.error("Error message")
    print("\n[OK] Logger configured successfully")
