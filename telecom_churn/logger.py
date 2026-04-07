import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import LOG_DIR, LOG_FILE


class LoggerFactory:
    @classmethod
    def get_logger(cls, name: str = __name__, level: int = logging.INFO) -> logging.Logger:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(name)
        if logger.handlers:
            return logger

        logger.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger
