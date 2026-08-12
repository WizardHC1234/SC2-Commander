import sys

import sc2

# TODO: Remove this after sk-learn no longer throws warnings with numpy version.
import warnings

warnings.filterwarnings("ignore")


class NonSharpyFilter:
    def __init__(self):
        self.module_name = "sharpy"

    def __call__(self, record):
        return not record["name"].startswith(self.module_name)


LOG_FORMAT = "<bold><green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <w>{message}</w></bold>"
# File sinks stay plain text (no markup / ANSI) for readable match logs.
LOG_FORMAT_FILE = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}"


class LoggingUtility:
    @staticmethod
    def set_logger_file(log_level: str, path: str):
        LoggingUtility.set_logger(log_level)
        root_logger = sc2.main.logger
        custom_filter = NonSharpyFilter()

        root_logger.add(
            path,
            format=LOG_FORMAT_FILE,
            level=log_level,
            filter="sharpy",
            colorize=False,
        )
        root_logger.add(
            path,
            format=LOG_FORMAT_FILE,
            level=log_level,
            filter=custom_filter,
            colorize=False,
        )

    @staticmethod
    def set_logger(log_level: str):
        root_logger = sc2.main.logger
        root_logger.remove()

        custom_filter = NonSharpyFilter()
        root_logger.add(sys.stderr, format=LOG_FORMAT, level=log_level, filter="sharpy")

        root_logger.add(sys.stderr, level=log_level, filter=custom_filter)

    @staticmethod
    def clear_logger():
        root_logger = sc2.main.logger
        root_logger.remove()
