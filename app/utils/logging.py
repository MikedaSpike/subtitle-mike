import logging
import sys

def setup_logger():
    logger = logging.getLogger("subtitle-mike")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False
    return logger