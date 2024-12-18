# src/utils/logger.py
from __future__ import annotations

import logging

from colorama import Back, Fore, Style, init

init(autoreset=True)


LEVEL_COLORS = {
    "DEBUG": Back.BLUE + Fore.WHITE,
    "INFO": Back.GREEN + Fore.BLACK,
    "WARNING": Back.YELLOW + Fore.BLACK,
    "ERROR": Back.RED + Fore.WHITE,
    "CRITICAL": Back.MAGENTA + Fore.WHITE,
}
TIME_COLOR = Fore.CYAN
MODULE_COLOR = Fore.MAGENTA

LEVEL_LENGTH = 8


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        level_color = LEVEL_COLORS.get(levelname, "")
        padded_levelname = record.levelname.ljust(LEVEL_LENGTH)
        record.levelname = f"{level_color} {padded_levelname}{Style.RESET_ALL}"
        base_message = super().format(record)

        allowed_keys = {"pathname", "lineno", "module"}
        extra_info = {k: v for k, v in vars(record).items() if k in allowed_keys}
        if extra_info and record.levelno != logging.INFO:
            extra_str = "\n"
            extra_str += f"{record.msg}\n"
            extra_str += f"Module: {record.module}\n"
            extra_str += f"Function: {record.funcName}\n"
            extra_str += f"on {record.pathname}#{record.lineno}\n"
            extra_str += f"Thread: {record.threadName} ({record.thread})\n"
            return f"{base_message}{extra_str}"
        return base_message


def setup_logger(name: str, key: str | None) -> logging.Logger:
    logger = logging.getLogger(f"{name}#{key}")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    format_str = (
        f"%(levelname)s|{TIME_COLOR}%(asctime)s{Style.RESET_ALL}| "
        f"[{MODULE_COLOR}{name}{f'#{key}' if key else ''}{Style.RESET_ALL}] "
        f"%(message)s"
    )

    date_format = "%H:%M:%S"

    handler.setFormatter(ColorFormatter(format_str, datefmt=date_format))
    logger.handlers = []
    logger.addHandler(handler)
    return logger
