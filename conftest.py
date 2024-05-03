import logging

import matplotlib
import pytest

logging.getLogger("matplotlib").setLevel(logging.WARNING)
# doesn't work because ffcx changes the log-level at runtime, which is not good.
logging.getLogger("ffcx").setLevel(logging.INFO)


def pytest_addoption(parser):
    parser.addoption("--hide-plots", action="store_true")


def pytest_configure(config):
    if config.getoption("hide_plots"):
        matplotlib.use("agg")


@pytest.fixture(autouse=True)
def _tracer():
    import saltx.trace

    saltx.trace.tracer.reset()
