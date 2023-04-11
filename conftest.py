import logging

import matplotlib

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def pytest_addoption(parser):
    parser.addoption("--hide-plots", action="store_true")


def pytest_configure(config):
    if config.getoption("hide_plots"):
        matplotlib.use("agg")
