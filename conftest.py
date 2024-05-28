import logging
from pathlib import Path

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


class _Infra:
    def __init__(self, node):
        self._node = node

    def save_plot(self, fig, name=None):
        # see https://stackoverflow.com/a/68804077/537149
        info = f"{self._node.path.name}\n{self._node.name}"

        fig.text(
            0.025,
            0.975,
            info,
            color="blue",
            fontsize="small",
            horizontalalignment="left",
            verticalalignment="top",
        )

        # Ensure the 'plots' directory exists
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)

        # Generate the filename
        if name:
            fname = plots_dir / f"{self._node.name}_{name}.png"
        else:
            fname = plots_dir / f"{self._node.name}.png"

        fig.savefig(fname)


@pytest.fixture()
def infra(request):

    return _Infra(request.node)
