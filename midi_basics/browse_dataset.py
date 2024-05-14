import yaml
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from datasets import Dataset, load_dataset


@st.cache_data
def load_hf_dataset(dataset_name: str, split: str):
    return load_dataset(dataset_name, split=split)


def select_dataset():
    dataset_names = ["roszcz/maestro-sustain-v2"]
    dataset_name = st.selectbox(label="dataset", options=dataset_names)
    split = st.text_input(label="split", value="train+test+validation")
    dataset = load_hf_dataset(dataset_name=dataset_name, split=split)
    return dataset


def select_record(midi_dataset: Dataset):
    source_df = midi_dataset.to_pandas()
    source_df["source"] = source_df["source"].map(lambda source: yaml.safe_load(source))
    source_df["composer"] = [source["composer"] for source in source_df.source]
    source_df["title"] = [source["title"] for source in source_df.source]

    composers = source_df.composer.unique()

    selected_composer = st.selectbox(
        label="Select composer",
        options=composers,
        index=3,
    )

    ids = source_df.composer == selected_composer
    piece_titles = source_df[ids].title.unique()
    selected_title = st.selectbox(
        label="Select title",
        options=piece_titles,
    )
    st.write(selected_title)

    ids = (source_df.composer == selected_composer) & (source_df.title == selected_title)
    part_df = source_df[ids]
    part_dataset = midi_dataset.select(part_df.index.values)

    return part_dataset[0]


def main():
    midi_dataset = select_dataset()

    record = select_record(midi_dataset=midi_dataset)

    piece = ff.MidiPiece.from_huggingface(record=record)
    st.json(piece.source)
    streamlit_pianoroll.from_fortepyan(piece=piece)


if __name__ == "__main__":
    main()
