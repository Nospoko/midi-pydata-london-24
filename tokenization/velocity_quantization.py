import numpy as np
import pandas as pd
import fortepyan as ff
import matplotlib.pyplot as plt
from datasets import load_dataset


def load_and_prepare_data() -> tuple[list[ff.MidiPiece], dict]:
    dataset = load_dataset("roszcz/maestro-sustain-v2")
    train_dataset = dataset["train"]
    pieces = [ff.MidiPiece.from_huggingface(record=record) for record in train_dataset]
    return pieces, dataset["test"]


def find_dataset_velocity_bin_edges(pieces: list[ff.MidiPiece], n_bins: int = 3) -> np.ndarray:
    velocities = np.hstack([p.df.velocity.values for p in pieces])
    quantiles = np.linspace(0, 1, num=n_bins + 1)
    bin_edges = np.quantile(velocities, quantiles)
    bin_edges[0] = 0
    bin_edges[-1] = 128
    return bin_edges


def create_figure() -> tuple[plt.Figure, plt.Axes]:
    width_px = 1920
    height_px = 1080
    dpi = 120
    fig_width = width_px / dpi
    fig_height = height_px / dpi
    figsize = (fig_width, fig_height)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    return fig, ax


def plot_original_velocity_data(df: pd.DataFrame) -> None:
    fig, ax = create_figure()
    ax.scatter(df["start"], df["velocity"], color="blue")
    ax.set_xlabel("Note Start Time")
    ax.set_ylabel("Velocity")
    ax.set_title("Original Velocity Data")
    ax.set_ylim(0, 128)
    plt.show()


def plot_velocity_data_with_bin_edges(df: pd.DataFrame, velocity_bin_edges: np.ndarray) -> None:
    fig, ax = create_figure()
    ax.scatter(df["start"], df["velocity"], color="blue")
    for edge in velocity_bin_edges:
        ax.axhline(y=edge, color="gray", linestyle="--")
    ax.set_xlabel("Note Start Time")
    ax.set_ylabel("Velocity")
    ax.set_title("Velocity Data with Bin Edges")
    ax.set_ylim(0, 128)
    plt.show()


def plot_binned_velocity_data(df: pd.DataFrame) -> None:
    fig, ax = create_figure()
    ax.scatter(df["start"], df["velocity_bin"], color="blue", alpha=0.6)
    for bin in range(5):
        ax.axhline(y=bin, color="gray", linestyle="--")
    ax.set_xlabel("Note Start Time")
    ax.set_ylabel("Quantized Velocity Bin")
    ax.set_title("Quantized Velocity Data")
    plt.show()


def main():
    pieces, test_dataset = load_and_prepare_data()
    velocity_bin_edges = find_dataset_velocity_bin_edges(
        pieces=pieces,
        n_bins=5,
    )
    print("Velocity Bin Edges:", velocity_bin_edges)

    # Select a specific record from the test dataset
    record = test_dataset[77]
    df = pd.DataFrame(record["notes"])

    # Plot the original velocity data
    plot_original_velocity_data(df)

    # Add horizontal lines for bin edges and plot
    plot_velocity_data_with_bin_edges(
        df=df,
        velocity_bin_edges=velocity_bin_edges,
    )

    # Digitize the velocity data
    df["velocity_bin"] = (
        np.digitize(
            x=df["velocity"],
            bins=velocity_bin_edges,
        )
        - 1
    )

    # Display the first few rows to check the quantized values
    print(df.head())

    # Plot the quantized velocity data
    plot_binned_velocity_data(df)


if __name__ == "__main__":
    main()
