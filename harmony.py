import numpy as np
import matplotlib.pyplot as plt


def main():
    width_px = 1920
    height_px = 1080

    dpi = 120

    fig_width = width_px / dpi
    fig_height = height_px / dpi
    figsize = (fig_width, fig_height)

    # Define the fundamental frequency
    fundamental_frequency = 440  # in Hz

    # Define the time range
    t = np.linspace(0, 0.005, 1000)  # 5 ms time range

    # Number of harmonics to display
    num_harmonics = 6

    colors = plt.cm.viridis(np.linspace(0, 1, num_harmonics))
    fig, ax = plt.subplots(num_harmonics, 1, figsize=figsize, dpi=dpi, sharex=True)
    period = 1 / fundamental_frequency

    # Plot the harmonics
    for n in range(1, num_harmonics + 1):
        frequency = fundamental_frequency * n
        y = np.sin(2 * np.pi * frequency * t)

        ax[n - 1].plot(
            t,
            y,
            color=colors[n - 1],
            linewidth=2,
            label=f"{n}th Harmonic: {frequency} Hz",
        )
        ax[n - 1].set_ylabel(
            "Amplitude",
            fontsize=12,
        )
        ax[n - 1].legend(
            fontsize=10,
            loc="upper right",
        )
        ax[n - 1].grid(
            True,
            which="both",
            linestyle="--",
            linewidth=0.5,
        )
        ax[n - 1].set_title(
            f"Harmonic {n}",
            fontsize=14,
            pad=10,
        )

        # Add vertical lines at multiples of the period of the fundamental frequency
        for i in range(1, int(t[-1] / period) + 1):
            ax[n - 1].axvline(x=i * period, color="red", linestyle="--", linewidth=1)

    fig.suptitle(
        "Harmonic Series of a Vibrating String",
        fontsize=20,
        weight="bold",
    )
    fig.text(
        0.5,
        0.04,
        "Time (s)",
        ha="center",
        fontsize=16,
    )
    fig.text(
        0.04,
        0.5,
        "Amplitude",
        va="center",
        rotation="vertical",
        fontsize=16,
    )

    plt.subplots_adjust(hspace=0.5)

    fig.savefig("data/img/harmonics.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
