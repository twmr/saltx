from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

reference_test_dir = Path(__file__).parent


def main():
    fig, ax = plt.subplots()

    fname1 = "ep_mode1.csv"
    fname2 = "ep_mode2.csv"

    mode1 = np.loadtxt(reference_test_dir / fname1, delimiter=",")
    mode2 = np.loadtxt(reference_test_dir / fname2, delimiter=",")

    ax.plot(
        mode1[:, 0],
        mode1[:, 1],
        "ro",
        alpha=0.9,
        label="mode1",
    )
    ax.plot(
        mode2[:, 0],
        mode2[:, 1],
        "ko",
        alpha=0.9,
        label="mode2",
    )

    ax.grid(True)
    ax.set_xlabel("d")
    ax.set_ylabel("intens")
    fig.legend()

    plt.show()


if __name__ == "__main__":
    main()
