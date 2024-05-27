import yaml
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datasets import load_dataset


def load_and_prepare_dataset():
    """
    Load the MIDI dataset and prepare the data for analysis.
    """
    dataset_path = "roszcz/maestro-sustain-v2"
    midi_dataset = load_dataset(
        path=dataset_path,
        split="train+test+validation",
    )

    # Convert dataset to pandas DataFrame for filtering
    source_df = midi_dataset.to_pandas()

    # Load metadata from string to dict for each record
    source_df["source"] = source_df["source"].map(lambda source: yaml.safe_load(source))
    source_df["composer"] = [source["composer"] for source in source_df.source]

    return midi_dataset, source_df


def plot_duration_distribution(
    first_composer_notes: np.ndarray,
    second_composer_notes: np.ndarray,
    composer1: str,
    composer2: str,
):
    """
    Plot the duration distribution of notes for two composers.
    """
    # Sample data to improve plot readability
    sample_size = min(len(first_composer_notes), len(second_composer_notes))
    sample_size = int(0.9 * sample_size)
    first_composer_notes = first_composer_notes.sample(sample_size)
    second_composer_notes = second_composer_notes.sample(sample_size)

    # Plotting
    width_px = 1920
    height_px = 1080
    dpi = 120

    fig_width = width_px / dpi
    fig_height = height_px / dpi
    figsize = (fig_width, fig_height)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot distributions
    bins = np.linspace(0, 5, 1000)
    ax.set_title("Duration distribution")
    ax.hist(
        first_composer_notes.duration,
        bins=bins,
        color="turquoise",
        label=composer1,
    )
    ax.hist(
        second_composer_notes.duration,
        alpha=0.6,
        bins=bins,
        color="orange",
        label=composer2,
    )
    ax.set_xlim(0, 2)  # Limit x-axis for better visualization

    fig.subplots_adjust(bottom=0.3, wspace=0.33)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))

    return fig, ax


def main():
    st.write("Select two composers from the dropdown menu to compare their note duration distributions.")

    # Load and prepare dataset
    midi_dataset, source_df = load_and_prepare_dataset()

    # List of unique composers in the dataset
    unique_composers = sorted(source_df["composer"].unique())

    composer1 = st.selectbox(
        "Select first composer:",
        unique_composers,
        index=unique_composers.index("Ludwig van Beethoven"),
    )
    composer2 = st.selectbox(
        "Select second composer:",
        unique_composers,
        index=unique_composers.index("Frédéric Chopin"),
    )

    composer_dataframes = {
        composer1: [],
        composer2: [],
    }

    # Select records based on composers
    for composer in [composer1, composer2]:
        idx = source_df.composer == composer
        part_df = source_df[idx]
        part_dataset = midi_dataset.select(part_df.index.values)

        for record in part_dataset:
            # Load notes of selected records
            df = pd.DataFrame(record["notes"])
            composer_dataframes[composer].append(df)

    # Concatenate notes dataframes for each composer
    first_composer_notes = pd.concat(composer_dataframes[composer1])
    second_composer_notes = pd.concat(composer_dataframes[composer2])

    # Plot duration distribution
    fig, ax = plot_duration_distribution(
        first_composer_notes=first_composer_notes,
        second_composer_notes=second_composer_notes,
        composer1=composer1,
        composer2=composer2,
    )
    st.pyplot(fig=fig)


if __name__ == "__main__":
    main()
