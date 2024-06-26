import re
import json
import uuid
import base64

import yaml
import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from datasets import Dataset, load_dataset


def download_button(object_to_download, download_filename, button_text):
    """
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """
    if isinstance(object_to_download, bytes):
        pass

    elif isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # Try JSON encode for everything else
    else:
        object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub(r"\d+", "", button_uuid)

    custom_css = f"""
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }}
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    a_html = f"""
    <a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">
        {button_text}
    </a>
    <br></br>
    """
    button_html = custom_css + a_html

    return button_html


@st.cache_data
def load_hf_dataset(dataset_name: str, split: str):
    return load_dataset(dataset_name, split=split)


def select_dataset():
    dataset_names = ["roszcz/maestro-sustain-v2"]
    dataset_name = st.selectbox(label="dataset", options=dataset_names)
    split = st.text_input(label="split", value="train+test+validation")
    dataset = load_hf_dataset(dataset_name=dataset_name, split=split)
    return dataset


def select_record(midi_dataset: Dataset, key: int = 0):
    source_df = midi_dataset.to_pandas()
    source_df["source"] = source_df["source"].map(lambda source: yaml.safe_load(source))
    source_df["composer"] = [source["composer"] for source in source_df.source]
    source_df["title"] = [source["title"] for source in source_df.source]

    composers = source_df.composer.unique()

    selected_composer = st.selectbox(
        label="Select composer",
        options=composers,
        index=1,
        key=f"select_composer_{key}",
    )

    ids = source_df.composer == selected_composer
    piece_titles = source_df[ids].title.unique()
    selected_title = st.selectbox(
        label="Select title",
        options=piece_titles,
        key=f"select_title_{key}",
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

    path = "tmp.mid"
    piece.to_midi().write(path)
    with open(path, "rb") as file:
        button_str = download_button(
            object_to_download=file.read(),
            download_filename="piece.mid",
            button_text="download selected piece",
        )
        st.markdown(button_str, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
