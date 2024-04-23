# Copyright (C) 2023 Thomas Hisch
#
# This file is part of saltx (https://github.com/thisch/saltx)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
from __future__ import annotations

import logging
import re
import sys
import time
from collections.abc import Mapping

log = logging.getLogger(__name__)

# DEFAULT_LOG_FORMAT = "%(levelname)-8s %(name)s:%(filename)s:%(lineno)d %(message)s"
DEFAULT_LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d | %(name)13.13s %(levelname)-8s %(message)s"
)

DEFAULT_LOG_DATE_FORMAT = "%H:%M:%S"


class Timer:
    __slots__ = (
        "msg",
        "stream",
        "start",
        "elapsed",
    )

    def __init__(self, stream, msg=None):
        self.msg = msg
        self.stream = stream

    def __enter__(self):
        self.start = time.monotonic()
        return self

    def __exit__(self, *args):
        end = time.monotonic()
        self.elapsed = end - self.start
        if self.msg:
            self.stream(self.msg + f" - took {self.elapsed*1e3:.1f} ms")
        else:
            self.stream(f"elapsed time: {self.elapsed*1e3:.1f} ms")


def main():
    log.info("ABC info")
    log.debug("ABC debug")
    log.error("ABC error")
    log.critical("ABC crit")


def colorize(text, color):
    return f"\x1b[{color}m{text}\x1b[00m"


# COLORS = {k: colorize("|%9s | " % v, LOGCOLORS[k]) for k, v in LOGBASE.items()}


# def markup(name, opts):
#     return colorize(name, list(opts)[0])


class ColoredLevelFormatter(logging.Formatter):
    """A logging formatter which colorizes the %(levelname)..s part of the log
    format passed to __init__."""

    LOGLEVEL_COLOROPTS: Mapping[int, str] = {
        logging.CRITICAL: "31",  # critical/fatal
        logging.ERROR: "31;01",
        logging.WARNING: "33",
        logging.WARN: "33",
        logging.INFO: "32",
        logging.DEBUG: "35",
        logging.NOTSET: "0",
    }

    LEVELNAME_FMT_REGEX = re.compile(r"%\(levelname\)([+-.]?\d*(?:\.\d+)?s)")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._original_fmt = self._style._fmt
        self._level_to_fmt_mapping: dict[int, str] = {}

        for level, color in self.LOGLEVEL_COLOROPTS.items():
            self.add_color_level(level, color)

    def add_color_level(self, level: int, color_opts: str) -> None:
        """Add or update color opts for a log level.

        :param level:
            Log level to apply a style to, e.g. ``logging.INFO``.
        :param color_opts:
            ANSI escape sequence color options. Capitalized colors indicates
            background color, i.e. ``'green', 'Yellow', 'bold'`` will give bold
            green text on yellow background.
        .. warning::
            This is an experimental API.
        """
        assert self._fmt is not None
        levelname_fmt_match = self.LEVELNAME_FMT_REGEX.search(self._fmt)
        if not levelname_fmt_match:
            return
        levelname_fmt = levelname_fmt_match.group()

        formatted_levelname = levelname_fmt % {"levelname": logging.getLevelName(level)}

        # add ANSI escape sequences around the formatted levelname
        colorized_formatted_levelname = colorize(
            "|%09s | " % formatted_levelname, color_opts
        )
        self._level_to_fmt_mapping[level] = self.LEVELNAME_FMT_REGEX.sub(
            colorized_formatted_levelname, self._fmt
        )

    def format(self, record: logging.LogRecord) -> str:
        fmt = self._level_to_fmt_mapping.get(record.levelno, self._original_fmt)
        self._style._fmt = fmt
        return super().format(record)


def setuplogging():
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    log_format = DEFAULT_LOG_FORMAT
    log_date_format = DEFAULT_LOG_DATE_FORMAT
    formatter: logging.Formatter = ColoredLevelFormatter(log_format, log_date_format)

    if logger.handlers:
        sh = logger.handlers[0]
    else:
        sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)


if __name__ == "__main__":
    setuplogging()

    main()
