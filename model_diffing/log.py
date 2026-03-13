import logging


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("model_diffing")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = setup_logger()
