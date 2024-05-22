import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from datasets import load_dataset

from tokenization.tokenization import NoLossTokenizer


@st.cache_data
def load_hf_dataset():
    dataset = load_dataset("roszcz/maestro-sustain-v2", split="test")
    return dataset


dataset = load_hf_dataset()


@st.cache_data
def prepare_piece(idx: int = 77):
    record = dataset[idx]
    piece = ff.MidiPiece.from_huggingface(record=record)
    return piece


piece = prepare_piece(77)


@st.cache_data
def prepare_tokens():
    path = "data/midi/example.mid"
    piece = ff.MidiPiece.from_file(path=path)
    tokenizer = NoLossTokenizer()
    tokens = tokenizer.tokenize(notes=piece.df)

    untokenized_notes = tokenizer.untokenize(tokens=tokens)
    untokenized_piece = ff.MidiPiece(untokenized_notes)
    return untokenized_piece, tokens


untokenized_piece, tokens = prepare_tokens()

maestro_description = pd.DataFrame(
    {
        "Split": ["Train", "Validation", "Test", "Total"],
        "Performances": [962, 137, 177, 1276],
        "Duration (hours)": [159.417374, 19.462732, 20.026704, 198.906810],
        "Notes (millions)": [5.659329, 0.639425, 0.741410, 7.040164],
    }
)


# Initialize session state
if "slide" not in st.session_state:
    st.session_state.slide = 0


# Slide content
slides = [
    # Opening slide
    {
        "images": ["data/img/cover.png"],
    },
    # Harmony
    {
        "header": '"All nature consists of harmony arising out of numbers"',
        "images": ["data/img/harmonics.png"],
    },
    # Hiller
    {
        "images": ["data/img/hiller.jpg"],
    },
    # Piano
    {
        "images": ["data/img/piano.jpg"],
    },
    # Scores
    {
        "images": ["data/img/scores.png"],
    },
    # Scored piece
    {"images": ["data/img/scores.png"], "piece_paths": ["data/midi/scored_piece.mid"]},
    # piano performance
    {
        "video": "data/Yuja_Wang.mp4",
    },
    # Spectrogram solo
    {"header": "Spectrograms vs. MIDI", "images": ["data/img/spectrogram.png"]},
    # Spectrograms vs. MIDI
    {
        "images": ["data/img/spectrogram.png", "data/img/pianoroll.png"],
    },
    # Yuja Wang in midi by Basic Pitch
    {
        "header": "Yuja Wang transcribed by Basic Pitch",
        "video": "data/Yuja_Wang.mp4",
        "piece_paths": ["data/midi/yuja_wang.mid"],
    },
    # MIDI sequence
    {
        "header": "Musical Instrument Digital Interface",
        "content": """
        ```plaintext
        0, 0, Header, 1, 1, 480
        1, 0, Start_track
        1, 0, Tempo, 500000
        1, 0, Time_signature, 4, 2, 24, 8
        1, 0, Program_c, 0, 0
        1, 0, Note_on_c, 0, 60, 127
        1, 480, Note_off_c, 0, 60, 0
        1, 480, Note_on_c, 0, 62, 127
        1, 960, Note_off_c, 0, 62, 0
        1, 960, Note_on_c, 0, 64, 127
        1, 1440, Note_off_c, 0, 64, 0
        1, 1440, Program_c, 0, 41
        1, 1440, Note_on_c, 0, 55, 127
        1, 1920, Note_off_c, 0, 55, 0
        1, 1920, Note_on_c, 0, 57, 127
        1, 2400, Note_off_c, 0, 57, 0
        1, 2400, Note_on_c, 0, 59, 127
        1, 2880, Note_off_c, 0, 59, 0
        1, 2880, End_track
        0, 0, End_of_file
        ```
        """,
    },
    # MIDI to DataFrame Conversion
    {
        "header": "Converting MIDI to DataFrame",
        "code": """
        import fortepyan as ff

        piece = ff.MidiPiece.from_file(path="data/midi/piano.mid")
        print(piece.df)
        """,
        "dataframe": piece.df,
    },
    # MIDI datasets
    {
        "header": "roszcz/maestro-sustain-v2",
        "dataframe": maestro_description,
        "content": """
        ```
        DatasetDict({
            train: Dataset({
                features: ['notes', 'source'],
                num_rows: 962
            })
            validation: Dataset({
                features: ['notes', 'source'],
                num_rows: 137
            })
            test: Dataset({
                features: ['notes', 'source'],
                num_rows: 177
            })
        })
        ```
        """,
        "code": """
        from datasets import load_dataset

        dataset = load_dataset("roszcz/maestro-sustain-v2")
        """,
    },
    # Visualizing and Listening to MIDI Files
    {
        "header": "Visualizing and Listening to MIDI Files",
        "code": """
        import fortepyan as ff
        import streamit_pianoroll
        from datasets import load_dataset

        dataset = load_dataset("roszcz/maestro-sustain-v2", split="test")
        idx = st.number_input(label="record id", value=77)
        record = dataset[idx]

        piece = ff.MidiPiece.from_huggingface(record=record)
        streamlit_pianoroll.from_fortepyan(piece=piece)
        """,
        "pieces": [piece],
    },
    # dstart
    {
        "code": 'piece.df["dstart"] = piece.df.start.diff().shift(-1)',
        "images": ["data/img/dstart_comparison.png"],
    },
    # duration
    {
        "images": ["data/img/duration_comparison.png"],
    },
    # pitch
    {
        "images": ["data/img/pitch_comparison.png"],
    },
    # quantization
    {
        "header": "Quantisation",
        "content": """
        ```py
        ["74-1-4-4", "71-0-4-4" "83-0-4-4" "79-0-4-4" "77-3-4-4"]
        ```
        """,
        "piece_paths": ["data/midi/example.mid", "data/midi/quantized_example.mid"],
    },
    # NoLossTokenizer
    {
        "pieces": [piece, untokenized_piece],
        "code": tokens[:20],
    },
    # BPE
    {
        "images": ["data/img/bpe.png"],
    },
    # links
    {
        "content":
    """
    Maestro dataset: https://magenta.tensorflow.org/datasets/maestro
    Github: https://github.com/Nospoko
    My Github: https://github.com/WojciechMat
    Presentation repo: https://github.com/Nospoko/midi-pydata-london-24
    Email: wmatejuk14@gmail.com
    """
    }
]


def main():
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    if col1.button("Previous"):
        if st.session_state.slide > 0:
            st.session_state.slide -= 1
    if col3.button("Next"):
        if st.session_state.slide < len(slides) - 1:
            st.session_state.slide += 1

    # Display the current slide
    slide = slides[st.session_state.slide]

    if "header" in slide:
        st.header(slide["header"])
        # Make Visualisation slide responsive
        if slide["header"] == "Visualizing and Listening to MIDI Files":
            st.code(slide["code"], language="python")
            idx = st.number_input(label="record id", value=77)
            record = dataset[idx]
            piece = ff.MidiPiece.from_huggingface(record=record)
            st.write(piece.source)
            streamlit_pianoroll.from_fortepyan(piece=piece)
            return
    if "code" in slide:
        st.code(slide["code"], language="python")
    if "content" in slide:
        st.write(slide["content"])
    if "dataframe" in slide:
        st.write(slide["dataframe"])
    if "images" in slide:
        for image in slide["images"]:
            st.image(image=image)
    if "video" in slide:
        st.video(slide["video"])
    if "pieces" in slide:
        for piece in slide["pieces"]:
            streamlit_pianoroll.from_fortepyan(piece=piece)
    if "piece_paths" in slide:
        for piece_path in slide["piece_paths"]:
            prepared_piece = ff.MidiPiece.from_file(piece_path)
            streamlit_pianoroll.from_fortepyan(piece=prepared_piece)


if __name__ == "__main__":
    main()
