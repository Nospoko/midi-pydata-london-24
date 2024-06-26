import numpy as np
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
import matplotlib.pyplot as plt

from browse_dataset import select_record, select_dataset


def plot_histogram(
    data1: np.ndarray,
    data2: np.ndarray,
    label1: str,
    label2: str,
    xlabel: str,
    ylabel: str,
    title: str,
    bins: np.ndarray,
) -> tuple[plt.Figure, plt.Axes]:
    """Utility function to plot histograms for comparison."""
    width_px = 1920
    height_px = 1080

    dpi = 120

    fig_width = width_px / dpi
    fig_height = height_px / dpi
    figsize = (fig_width, fig_height)

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi,
    )
    ax.hist(
        data1,
        bins=bins,
        color="turquoise",
        label=label1,
    )
    ax.hist(
        data2,
        bins=bins,
        alpha=0.6,
        color="orange",
        label=label2,
    )
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        fontsize=12,
    )
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    return fig, ax


def main() -> None:
    dataset = select_dataset()
    select_columns = st.columns(2)

    with select_columns[0]:
        first_record = select_record(midi_dataset=dataset, key=0)
    with select_columns[1]:
        second_record = select_record(midi_dataset=dataset, key=1)

    first_piece = ff.MidiPiece.from_huggingface(first_record)
    second_piece = ff.MidiPiece.from_huggingface(second_record)

    def get_label(piece: ff.MidiPiece) -> str:
        composer = piece.source.get("composer", "Unknown Composer")
        title = piece.source.get("title", "Unknown Title")
        return f"{title}, {composer}"

    label1 = get_label(first_piece)
    label2 = get_label(second_piece)

    # Plot pitch comparison
    bins = np.linspace(21, 109, num=89)
    fig, ax = plot_histogram(
        data1=first_piece.df.pitch.values,
        data2=second_piece.df.pitch.values,
        label1=label1,
        label2=label2,
        xlabel="Pitch",
        ylabel="Frequency",
        title="Note Pitches",
        bins=bins,
    )
    st.pyplot(fig)
    # Calculate dstart for both pieces
    first_piece.df["dstart"] = first_piece.df.start.diff().shift(-1).fillna(0)
    second_piece.df["dstart"] = second_piece.df.start.diff().shift(-1).fillna(0)

    # Plot dstart comparison
    bins = np.linspace(0, 1.00, num=200)
    fig, ax = plot_histogram(
        data1=first_piece.df.dstart.fillna(0).values,
        data2=second_piece.df.dstart.fillna(0).values,
        label1=label1,
        label2=label2,
        xlabel="Time Difference (s)",
        ylabel="Frequency",
        title="Time Difference Distribution",
        bins=bins,
    )
    ax.set_xlim((0, 0.4))
    st.pyplot(fig)

    # Plot distributions
    bins = np.linspace(0, 5, 1000)
    fig, ax = plot_histogram(
        data1=first_piece.df.duration,
        data2=second_piece.df.duration,
        label1=label1,
        label2=label2,
        xlabel="Duration (s)",
        ylabel="Frequency",
        title="Note Duration Distribution",
        bins=bins,
    )
    ax.set_xlim(0, 1)  # Limit x-axis for better visualization

    fig.subplots_adjust(bottom=0.3, wspace=0.33)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
    st.pyplot(fig)

    bins = np.linspace(0, 127, 128, endpoint=True)
    fig, ax = plot_histogram(
        data1=first_piece.df.velocity,
        data2=second_piece.df.velocity,
        label1=label1,
        label2=label2,
        xlabel="Velocity",
        ylabel="Frequency",
        title="Key-press Velocity Distribution",
        bins=bins,
    )
    fig.subplots_adjust(bottom=0.3, wspace=0.33)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
    st.pyplot(fig)
    show_pianorolls = st.checkbox(
        "show pianorolls",
        value=False,
    )

    if not show_pianorolls:
        return

    streamlit_pianoroll.from_fortepyan(first_piece)
    streamlit_pianoroll.from_fortepyan(second_piece)


if __name__ == "__main__":
    main()
