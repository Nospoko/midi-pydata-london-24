import yaml
import fortepyan as ff
import matplotlib.pyplot as plt
from datasets import load_dataset
import numpy as np


def plot_histogram(data1, data2, label1, label2, xlabel, ylabel, title, bins=88):
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


def main():
    # Load the MIDI dataset
    dataset_path = "roszcz/maestro-sustain-v2"
    midi_dataset = load_dataset(path=dataset_path, split="train+test+validation")

    # Convert to DataFrame and process metadata
    source_df = midi_dataset.to_pandas()
    source_df["source"] = source_df["source"].apply(yaml.safe_load)
    source_df["midi_filename"] = source_df["source"].apply(lambda src: src["midi_filename"])

    pieces = []

    # Filenames of MIDI files from maestro dataset to be compared
    # Different pieces
    # filenames = [
    #     "2004/MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID--AUDIO_12_R1_2004_07_Track07_wav.midi",
    #     "2008/MIDI-Unprocessed_02_R3_2008_01-03_ORIG_MID--AUDIO_02_R3_2008_wav--2.midi",
    # ]
    # Different performances
    filenames = [
        "2011/MIDI-Unprocessed_15_R1_2011_MID--AUDIO_R1-D6_10_Track10_wav.midi",
        "2014/MIDI-UNPROCESSED_06-08_R1_2014_MID--AUDIO_07_R1_2014_wav--6.midi",
    ]

    for filename in filenames:
        idx = source_df["midi_filename"] == filename
        if idx.any():
            record = midi_dataset[idx.idxmax()]
            pieces.append(ff.MidiPiece.from_huggingface(record))

    if len(pieces) < 2:
        print("Not enough MIDI files found for comparison.")
        return

    first_piece, second_piece = pieces

    # Get truncated titles for labels
    def get_truncated_title(piece, max_length=30):
        title = piece.source.get("title", "Unknown Title")
        return title if len(title) <= max_length else title[:max_length] + "..."

    label1 = get_truncated_title(first_piece)
    label2 = get_truncated_title(second_piece)

    # Plot pitch comparison
    fig, ax = plot_histogram(
        data1=first_piece.df.pitch,
        data2=second_piece.df.pitch,
        label1=label1,
        label2=label2,
        xlabel="Pitch",
        ylabel="Frequency",
        title="Note Pitches",
    )

    # Calculate dstart for both pieces
    first_piece.df["dstart"] = first_piece.df.start.diff().shift(-1)
    second_piece.df["dstart"] = second_piece.df.start.diff().shift(-1)

    # Plot dstart comparison
    bins = np.linspace(0, max(first_piece.df.dstart.max(), second_piece.df.dstart.max()), num=200)
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
    main()
