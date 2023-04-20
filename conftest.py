import logging

import matplotlib

logging.getLogger("matplotlib").setLevel(logging.WARNING)
# doesn't work because ffcx changes the log-level at runtime, which is not good.
logging.getLogger("ffcx").setLevel(logging.INFO)


def pytest_addoption(parser):
    parser.addoption("--hide-plots", action="store_true")


def pytest_configure(config):
    if config.getoption("hide_plots"):
        matplotlib.use("agg")
