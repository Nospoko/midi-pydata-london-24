import numpy as np
import pandas as pd
import fortepyan as ff
import matplotlib.pyplot as plt
from datasets import load_dataset


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the Maestro Sustain V2 dataset.

    Returns:
        tuple: Training and test datasets as pandas DataFrames.
    """
    dataset = load_dataset("roszcz/maestro-sustain-v2")
    return dataset["train"], dataset["test"]


def prepare_pieces(
    train_dataset: pd.DataFrame,
) -> list[ff.MidiPiece]:
    """
    Convert training dataset records to MidiPiece objects.

    Args:
        train_dataset (pd.DataFrame): Training dataset.

    Returns:
        list: List of MidiPiece objects.
    """
    return [ff.MidiPiece.from_huggingface(record=record) for record in train_dataset]


def find_dataset_dstart_bin_edges(
    pieces: list[ff.MidiPiece],
    n_bins: int = 3,
) -> np.ndarray:
    """
    Calculate bin edges for dstart values across all pieces.

    Args:
        pieces (list): List of MidiPiece objects.
        n_bins (int): Number of bins to create.

    Returns:
        np.ndarray: Array of bin edges for dstart values.
    """
    dstarts = []
    for piece in pieces:
        next_start = piece.df.start.shift(-1)
        dstart = next_start - piece.df.start
        dstarts.append(dstart[:-1])

    quantiles = np.linspace(0, 1, num=n_bins)
    dstarts = np.hstack(dstarts)
    bin_edges = np.quantile(dstarts, quantiles)[:-1]
    bin_edges[0] = 0
    last_edge = max(bin_edges[-1] * 3, 0.5)
    bin_edges = np.append(bin_edges, last_edge)
    return bin_edges


def find_dataset_duration_bin_edges(
    pieces: list[ff.MidiPiece],
    n_bins: int = 3,
) -> np.ndarray:
    """
    Calculate bin edges for duration values across all pieces.

    Args:
        pieces (list): List of MidiPiece objects.
        n_bins (int): Number of bins to create.

    Returns:
        np.ndarray: Array of bin edges for duration values.
    """
    df = pd.concat([p.df for p in pieces])
    duration = df.duration.values
    quantiles = np.linspace(0, 1, num=n_bins + 1)
    bin_edges = np.quantile(duration, quantiles)[:-1]
    bin_edges[0] = 0
    return bin_edges


def find_dataset_velocity_bin_edges(
    pieces: list[ff.MidiPiece],
    n_bins: int = 3,
) -> np.ndarray:
    """
    Calculate bin edges for velocity values across all pieces.

    Args:
        pieces (list): List of MidiPiece objects.
        n_bins (int): Number of bins to create.

    Returns:
        np.ndarray: Array of bin edges for velocity values.
    """
    velocities = np.hstack([p.df.velocity.values for p in pieces])
    quantiles = np.linspace(0, 1, num=n_bins + 1)
    bin_edges = np.quantile(velocities, quantiles)
    bin_edges[0] = 0
    bin_edges[-1] = 128
    return bin_edges


def calculate_bins(
    df: pd.DataFrame,
    dstart_bin_edges: np.ndarray,
    duration_bin_edges: np.ndarray,
    velocity_bin_edges: np.ndarray,
) -> pd.DataFrame:
    """
    Digitize the dstart, duration, and velocity values into bins.

    Args:
        df (pd.DataFrame): DataFrame containing note data.
        dstart_bin_edges (np.ndarray): Bin edges for dstart values.
        duration_bin_edges (np.ndarray): Bin edges for duration values.
        velocity_bin_edges (np.ndarray): Bin edges for velocity values.

    Returns:
        pd.DataFrame: DataFrame with digitized values.
    """
    next_start = df.start.shift(-1)
    df["dstart"] = (next_start - df.start).fillna(0)

    df["velocity_bin"] = (
        np.digitize(
            x=df["velocity"],
            bins=velocity_bin_edges,
        )
        - 1
    )
    df["dstart_bin"] = (
        np.digitize(
            x=df["dstart"],
            bins=dstart_bin_edges,
        )
        - 1
    )
    df["duration_bin"] = (
        np.digitize(
            x=df["duration"],
            bins=duration_bin_edges,
        )
        - 1
    )
    return df


def dequantize_bins(
    df: pd.DataFrame,
    dstart_bin_edges: np.ndarray,
    duration_bin_edges: np.ndarray,
    velocity_bin_edges: np.ndarray,
) -> pd.DataFrame:
    """
    Dequantize the binned values back to their original range.

    Args:
        df (pd.DataFrame): DataFrame containing binned note data.
        dstart_bin_edges (np.ndarray): Bin edges for dstart values.
        duration_bin_edges (np.ndarray): Bin edges for duration values.
        velocity_bin_edges (np.ndarray): Bin edges for velocity values.

    Returns:
        pd.DataFrame: DataFrame with dequantized values.
    """
    # Calculate bin midpoints
    dstart_midpoints = (dstart_bin_edges[:-1] + dstart_bin_edges[1:]) / 2
    dstart_midpoints = np.append(dstart_midpoints, dstart_bin_edges[-1])
    duration_midpoints = (duration_bin_edges[:-1] + duration_bin_edges[1:]) / 2
    duration_midpoints = np.append(duration_midpoints, duration_bin_edges[-1])
    velocity_midpoints = (velocity_bin_edges[:-1] + velocity_bin_edges[1:]) / 2

    # Map bin indices to midpoints
    df["dstart_dequantized"] = dstart_midpoints[df["dstart_bin"]]
    df["duration_dequantized"] = duration_midpoints[df["duration_bin"]]
    df["velocity_dequantized"] = velocity_midpoints[df["velocity_bin"]]

    return df


def plot_data(
    df: pd.DataFrame,
    dstart_bin_edges: np.ndarray,
    duration_bin_edges: np.ndarray,
    velocity_bin_edges: np.ndarray,
) -> None:
    """
    Plot the original and digitized data with bin edges.

    Args:
        df (pd.DataFrame): DataFrame containing note data.
        dstart_bin_edges (np.ndarray): Bin edges for dstart values.
        duration_bin_edges (np.ndarray): Bin edges for duration values.
        velocity_bin_edges (np.ndarray): Bin edges for velocity values.
    """
    width_px = 1920
    height_px = 1080
    dpi = 120
    fig_width = width_px / dpi
    fig_height = height_px / dpi
    figsize = (fig_width, fig_height)

    fig, ax = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=figsize,
        dpi=dpi,
    )

    # Plot original velocity data
    ax[0].scatter(
        x=df["start"],
        y=df["velocity"],
        color="blue",
        label="start",
    )
    ax[0].set_xlabel("Note Start Time")
    ax[0].set_ylabel("Velocity")
    ax[0].set_title("Original Note Events")
    ax[0].legend()
    for edge in velocity_bin_edges:
        ax[0].axhline(
            y=edge,
            color="gray",
            linestyle="--",
        )

    # Plot calculated dstart data
    ax[1].scatter(
        x=df["start"],
        y=df["dstart"],
        color="green",
        label="dstart",
    )
    ax[1].set_xlabel("Note Start Time")
    ax[1].set_ylabel("dstart (s)")
    ax[1].set_title("Calculated dstart")
    ax[1].legend()
    for edge in dstart_bin_edges:
        ax[1].axhline(
            y=edge,
            color="gray",
            linestyle="--",
        )

    # Plot calculated duration data
    ax[2].scatter(
        x=df["start"],
        y=df["duration"],
        color="red",
        label="duration",
    )
    ax[2].set_xlabel("Note Start Time")
    ax[2].set_ylabel("Duration (s)")
    ax[2].set_title("Calculated Duration")
    ax[2].legend()
    for edge in duration_bin_edges:
        ax[2].axhline(
            y=edge,
            color="gray",
            linestyle="--",
        )

    # Plot quantized bin data
    ax[3].scatter(
        x=df["start"],
        y=df["velocity_bin"],
        color="blue",
        alpha=0.6,
        label="Velocity Bin",
    )
    ax[3].scatter(
        x=df["start"],
        y=df["dstart_bin"],
        color="green",
        alpha=0.6,
        label="dstart Bin",
    )
    ax[3].scatter(
        x=df["start"],
        y=df["duration_bin"],
        color="red",
        alpha=0.6,
        label="Duration Bin",
    )
    ax[3].set_xlabel("Note Start Time")
    ax[3].set_ylabel("Quantized Bins")
    ax[3].set_title("Quantized Values (5 bins each)")
    ax[3].legend()
    for edge in range(6):
        ax[3].axhline(
            y=edge,
            color="gray",
            linestyle="--",
        )

    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    Main function to load data, prepare MIDI pieces, calculate bin edges,
    and plot the original and quantized note data.
    """
    train_dataset, test_dataset = load_data()
    pieces = prepare_pieces(
        train_dataset=train_dataset,
    )

    # Find bin edges for dstart, duration, and velocity
    dstart_bin_edges = find_dataset_dstart_bin_edges(
        pieces=pieces,
        n_bins=5,
    )
    duration_bin_edges = find_dataset_duration_bin_edges(
        pieces=pieces,
        n_bins=5,
    )
    velocity_bin_edges = find_dataset_velocity_bin_edges(
        pieces=pieces,
        n_bins=5,
    )

    # Select a specific record from the test dataset
    record = test_dataset[77]
    df = pd.DataFrame(record["notes"])

    # Digitize the velocity, dstart, and duration data
    df = calculate_bins(
        df=df,
        dstart_bin_edges=dstart_bin_edges,
        duration_bin_edges=duration_bin_edges,
        velocity_bin_edges=velocity_bin_edges,
    )
    print(dstart_bin_edges)
    # Dequantize the binned data
    df = dequantize_bins(
        df=df,
        dstart_bin_edges=dstart_bin_edges,
        duration_bin_edges=duration_bin_edges,
        velocity_bin_edges=velocity_bin_edges,
    )

    # Plot the original and digitized data with bin edges
    plot_data(
        df=df,
        dstart_bin_edges=dstart_bin_edges,
        duration_bin_edges=duration_bin_edges,
        velocity_bin_edges=velocity_bin_edges,
    )


if __name__ == "__main__":
    main()
