import yaml
import fortepyan as ff
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from datasets import load_dataset
from matplotlib.figure import Figure

# Filenames of MIDI files from maestro dataset to be compared
filenames = [
    "2004/MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID--AUDIO_12_R1_2004_07_Track07_wav.midi",
    "2008/MIDI-Unprocessed_02_R3_2008_01-03_ORIG_MID--AUDIO_02_R3_2008_wav--2.midi",
]


def plot_pitch_comparison(first_piece: ff.MidiPiece, second_piece: ff.MidiPiece):
    """
    Plot comparison of note pitches between two MIDI pieces.

    Args:
    - first_piece: Instance of ff.MidiPiece representing the first MIDI piece.
    - second_piece: Instance of ff.MidiPiece representing the second MIDI piece.

    Returns:
    - fig: Matplotlib Figure object.
    - ax: Matplotlib Axes object.
    """

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots()
    # Metadata for legend
    title_first = first_piece.source["title"]
    title_second = second_piece.source["title"]
    composer_first = first_piece.source["composer"]
    composer_second = second_piece.source["composer"]

    # Plot distributions
    ax.set_title("Note pitches")
    ax.hist(first_piece.df.pitch, bins=88, color="turquoise", label=f"{composer_first}: {title_first}")
    ax.hist(second_piece.df.pitch, alpha=0.6, bins=88, color="orange", label=f"{composer_second}: {title_second}")

    fig.subplots_adjust(bottom=0.3, wspace=0.33)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
    return fig, ax


def plot_dstart_comparison(first_piece: ff.MidiPiece, second_piece: ff.MidiPiece):
    """
    Plot comparison of note start time differences (dstart) between two MIDI pieces.

    Args:
    - first_piece: Instance of ff.MidiPiece representing the first MIDI piece.
    - second_piece: Instance of ff.MidiPiece representing the second MIDI piece.

    Returns:
    - fig: Matplotlib Figure object.
    - ax: Matplotlib Axes object.
    """
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots()

    # Metadata for legend
    title_first = first_piece.source["title"]
    title_second = second_piece.source["title"]
    filename_first = first_piece.source["midi_filename"]
    filename_second = second_piece.source["midi_filename"]

    first_df = first_piece.df
    second_df = second_piece.df

    # Calculate dstarts
    first_df["next_start"] = first_df.start.shift(-1)
    first_df["dstart"] = first_df.next_start - first_df.start

    second_df["next_start"] = second_df.start.shift(-1)
    second_df["dstart"] = second_df.next_start - second_df.start

    # Plotting distributions
    ax.set_title("Dstart distribution")
    ax.hist(first_df.dstart, bins=200, color="turquoise", label=f"{title_first} ({filename_first})")
    ax.hist(second_df.dstart, alpha=0.6, bins=200, color="orange", label=f"{title_second} ({filename_second})")

    fig.subplots_adjust(bottom=0.3, wspace=0.33)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
    return fig, ax


def main():
    # Path to the MIDI dataset
    dataset_path = "roszcz/maestro-sustain-v2"
    midi_dataset = load_dataset(
        path=dataset_path,
        split="train+test+validation",
    )

    # Convert dataset to pandas DataFrame for filtering
    source_df = midi_dataset.to_pandas()

    # Load metadata from string to dict for each record
    source_df["source"] = source_df["source"].map(lambda source: yaml.safe_load(source))
    source_df["midi_filename"] = [source["midi_filename"] for source in source_df.source]

    pieces = []

    # Select MIDI pieces based on filenames
    for filename in filenames:
        idx = source_df.midi_filename == filename
        part_df = source_df[idx]
        part_dataset = midi_dataset.select(part_df.index.values)

        record = part_dataset[0]
        piece = ff.MidiPiece.from_huggingface(record)
        pieces.append(piece)

    first_piece = pieces[0]
    second_piece = pieces[1]

    # Plot pitch comparison between two MIDI pieces
    fig, ax = plot_pitch_comparison(
        first_piece=first_piece,
        second_piece=second_piece,
    )
    plt.show()


if __name__ == "__main__":
    main()
