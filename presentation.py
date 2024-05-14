import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from datasets import load_dataset


@st.cache_data
def load_hf_dataset():
    dataset = load_dataset("roszcz/maestro-sustain-v2", split="test")
    return dataset


dataset = load_hf_dataset()


@st.cache_data
def prepare_first_piece():
    record = dataset[77]
    piece = ff.MidiPiece.from_huggingface(record=record)
    return piece


piece = prepare_first_piece()

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
        "images": ["data/img/vibration_modes.png"],
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
    # Spectrograms vs. MIDI
    {
        "header": "Spectrograms vs. MIDI",
        "images": ["data/img/spectrogram.png", "data/img/pianoroll.png"],
    },
    # MIDI to DataFrame Conversion
    {
        "header": "Converting MIDI to DataFrame",
        "content": """
        Here is an example of converting a MIDI file to a DataFrame using the fortepyan library.
        """,
        "code": """
        import fortepyan as ff

        piece = ff.MidiPiece.from_file(path="data/piano.mid")
        print(piece.df)
        """,
        "dataframe": piece.df,
    },
    # MIDI datasets
    {
        "header": "maestro-sustain-v2",
        "dataframe": maestro_description,
    },
    # Visualizing and Listening to MIDI Files
    {
        "header": "Visualizing and Listening to MIDI Files",
        "code": """
        dataset = load_dataset("roszcz/maestro-sustain-v2", split="test")
        record = dataset[77]

        piece = ff.MidiPiece.from_huggingface(record=record)
        streamlit_pianoroll.from_fortepyan(piece=piece)
        """,
        "pieces": [piece],
    },
    # dstart
    {
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
]

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
if "content" in slide:
    st.write(slide["content"])
if "code" in slide:
    st.code(slide["code"], language="python")
if "dataframe" in slide:
    st.write(slide["dataframe"])
if "pieces" in slide:
    for piece in slide["pieces"]:
        streamlit_pianoroll.from_fortepyan(piece=piece)
if "images" in slide:
    for image in slide["images"]:
        st.image(image=image)
