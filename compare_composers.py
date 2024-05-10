import yaml
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from datasets import load_dataset
from matplotlib.figure import Figure

composers = ["Ludwig van Beethoven", "Frédéric Chopin"]


def main():
    dataset_path = "roszcz/maestro-sustain-v2"
    midi_dataset = load_dataset(
        path=dataset_path,
        split="train+test+validation",
    )

    source_df = midi_dataset.to_pandas()
    source_df["source"] = source_df["source"].map(lambda source: yaml.safe_load(source))
    source_df["composer"] = [source["composer"] for source in source_df.source]

    composer_dataframes = {
        composers[0]: [],
        composers[1]: [],
    }
    for composer in composers:
        idx = source_df.composer == composer
        part_df = source_df[idx]
        part_dataset = midi_dataset.select(part_df.index.values)

        for record in part_dataset:
            df = pd.DataFrame(record["notes"])
            composer_dataframes[composer].append(df)

    first_composer_notes = pd.concat(composer_dataframes[composers[0]])
    second_composer_notes = pd.concat(composer_dataframes[composers[1]])

    first_composer_notes = first_composer_notes.sample(50000)
    second_composer_notes = second_composer_notes.sample(50000)

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots()

    # Plotting distributions
    ax.set_title("Duration distribution")
    ax.hist(first_composer_notes.duration, bins=1000, color="turquoise", label=composers[0])
    ax.hist(second_composer_notes.duration, alpha=0.6, bins=1000, color="orange", label=composers[1])
    ax.set_xlim(0, 5)

    fig.subplots_adjust(bottom=0.3, wspace=0.33)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))

    plt.show()


if __name__ == "__main__":
    main()
