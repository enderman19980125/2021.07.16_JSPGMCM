import logging


class Logger:
    def __init__(self, logger_name: str, file_name: str):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')

        handler = logging.FileHandler(f"{file_name}", mode="a")
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def debug(self, msg) -> None:
        self.logger.debug(msg)

    def info(self, msg) -> None:
        self.logger.info(msg)

    def warning(self, msg) -> None:
        self.logger.warning(msg)

    def error(self, msg) -> None:
        self.logger.error(msg)

    def critical(self, msg) -> None:
        self.logger.critical(msg)
