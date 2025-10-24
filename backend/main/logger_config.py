import logging
from functools import wraps
import time

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Suppress noisy loggers
    noisy_loggers = ["httpx", "httpcore", "ollama", "langchain_ollama", "boto3", "botocore", "urllib3"]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)


def log_duration(name: str):
    """Decorator to log duration of a function call."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            logger_name = getattr(func, "__module__", "__main__")
            logger = logging.getLogger(logger_name)
            logger.debug("%s took %.2f sec", name, time.time() - start)
            return result
        return wrapper
    return decorator