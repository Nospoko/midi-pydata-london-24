import argparse

import numpy as np
import fortepyan as ff
import matplotlib.pyplot as plt


def plot_histogram(
    data1: np.ndarray,
    data2: np.ndarray,
    label1: str,
    label2: str,
    xlabel: str,
    ylabel: str,
    title: str,
    bins: np.ndarray,
):
    """Utility function to plot histograms for comparison."""
    width_px = 1920
    height_px = 1080
    dpi = 120
    fig_width = width_px / dpi
    fig_height = height_px / dpi
    figsize = (fig_width, fig_height)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.hist(data1, bins=bins, color="turquoise", label=label1)
    ax.hist(data2, bins=bins, alpha=0.6, color="orange", label=label2)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    return fig, ax


def main(args):
    # Load MIDI files
    pieces = []
    try:
        pieces.append(ff.MidiPiece.from_file(args.filepath1))
        pieces.append(ff.MidiPiece.from_file(args.filepath2))
    except Exception as e:
        print(f"Error loading MIDI files: {e}")
        return

    if len(pieces) < 2:
        print("Not enough MIDI files found for comparison.")
        return

    first_piece, second_piece = pieces
    import os

    label1 = os.path.basename(args.filepath1)
    label2 = os.path.basename(args.filepath2)

    # Plot pitch comparison
    bins = np.linspace(21, 109, num=89)
    plot_histogram(
        data1=first_piece.df.pitch,
        data2=second_piece.df.pitch,
        label1=label1,
        label2=label2,
        xlabel="Pitch",
        ylabel="Frequency",
        title="Note Pitches",
        bins=bins,
    )

    # Calculate dstart for both pieces
    first_piece.df["dstart"] = first_piece.df.start.diff().shift(-1)
    second_piece.df["dstart"] = second_piece.df.start.diff().shift(-1)

    # Plot dstart comparison
    bins = np.linspace(0, 1, num=200)
    fig, ax = plot_histogram(
        data1=first_piece.df.dstart.dropna(),
        data2=second_piece.df.dstart.dropna(),
        label1=label1,
        label2=label2,
        xlabel="Time Difference (s)",
        ylabel="Frequency",
        title="Dstart Distribution",
        bins=bins,
    )
    ax.set_xlim((0, 0.4))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two MIDI files.")
    parser.add_argument("filepath1", type=str, help="Path to the first MIDI file")
    parser.add_argument("filepath2", type=str, help="Path to the second MIDI file")
    args = parser.parse_args()
    main(args)
