import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from datasets import load_dataset

from modelling.augmentation import augmentation_review
from tokenization.tokenization import ExponentialTimeTokenizer


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


@st.cache_data
def prepare_generated_piece():
    path = "data/midi/mozart_with_gpt_08.mid"
    path_gen = "data/midi/only_generated_mozart.mid"
    dual_piece = ff.MidiPiece.from_file(path)
    generated_piece = ff.MidiPiece.from_file(path_gen)
    original_piece = dual_piece[: generated_piece.size]
    return original_piece, generated_piece


piece = prepare_piece(77)
original_piece, generated_piece = prepare_generated_piece()


@st.cache_data
def prepare_tokens():
    path = "data/midi/example.mid"
    piece = ff.MidiPiece.from_file(path=path)
    tokenizer = ExponentialTimeTokenizer()
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

exponential_time_tokens = {
    "1T": "10ms",
    "2T": "20ms",
    "3T": "40ms",
    "4T": "80ms",
    "5T": "160ms",
    "6T": "320ms",
    "7T": "640ms",
}


# Initialize session state
if "slide" not in st.session_state:
    st.session_state.slide = 0


# Slide content
slides = [
    # Opening slide
    {
        "images": ["data/img/cover.png"],
    },
    # links
    {
        "header": "Important links",
        "content": """
    Maestro dataset: https://magenta.tensorflow.org/datasets/maestro

    Github: https://github.com/Nospoko

    My Github: https://github.com/WojciechMat

    Presentation repo: https://github.com/Nospoko/midi-pydata-london-24

    Pianoroll: https://pianoroll.io/

    Email: wmatejuk14@gmail.com
    """,
    },
    # Pythagoreans
    {
        "header": '"There is geometry in the humming of the strings, and there is music in the spacing of the spheres"',
        "images": ["data/img/pythagoras.png"],
    },
    {
        "header": "Algorithmic music composition 1 ",
        "images": {"data/img/mozart_table.png"},
    },
    # Hiller
    {
        "header": "Algorithmic music composition 2",
        "images": ["data/img/hiller.jpg"],
    },
    # Cage
    {
        "header": "Algorithmic music composition",
        "images": ["data/img/cage.jpg"],
    },
    # Piano
    {
        "header": "Piano",
        "images": ["data/img/graphics3.jpg"],
    },
    # pedals
    {
        "header": "Pedals",
        "images": ["data/img/pedals.jpg"],
    },
    # Scores
    {
        "header": "Scores sheet 1",
        "images": ["data/img/scores.png"],
    },
    # Scored piece
    {
        "header": "Scores sheet 2",
        "images": ["data/img/scores.png"],
        "piece_paths": ["data/midi/scored_piece.mid"],
    },
    # piano performance
    {
        "header": "Yuja Wang - Flight of a Bublebee",
        "video": "data/Yuja_Wang.mp4",
    },
    # Spectrogram solo
    {
        "header": "Spectrograms vs. MIDI 1",
        "images": ["data/img/spectrogram.png"],
    },
    # Spectrograms vs. MIDI
    {
        "header": "Spectrograms vs MIDI 2",
        "images": [
            "data/img/spectrogram.png",
            "data/img/pianoroll.png",
        ],
    },
    # Yuja Wang in midi by Basic Pitch
    {
        "header": "Yuja Wang transcribed by Basic Pitch",
        # "video": "data/Yuja_Wang.mp4",
        "piece_paths": ["data/midi/yuja_wang.mid"],
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
        "header": "Visualising and Listening to MIDI Files",
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
    # duration
    {
        "header": "Duration distribution",
        "images": ["data/img/duration_comparison.png"],
    },
    # dstart
    {
        "header": "Dstart distribution",
        "code": 'piece.df["dstart"] = piece.df.start.diff().shift(-1)',
        "images": ["data/img/dstart_comparison.png"],
    },
    # pitch
    {
        "header": "Pitch distribution",
        "images": ["data/img/pitch_comparison.png"],
    },
    # Modelling
    {
        "header": "Modelling piano performances with Large Language Models",
    },
    # Augmentation
    {
        "header": "Augmentation",
        "content": """
            #### Pitch shifting
            ```py
            def pitch_shift(df: pd.DataFrame, shift: int = 5) -> pd.DataFrame:
                df.pitch += shift
                return df, shift
            ```
            #### Speed change
            ```py
            def change_speed(df: pd.DataFrame, factor: float = None) -> pd.DataFrame:
                df.start /= factor
                df.end /= factor
                df.duration = df.end - df.start
                return df
            """,
    },
    # Quantization
    {
        "header": "One token per note",
        "content": """
        ```py
        ["74-1-4-4", "71-0-4-4" "83-0-4-4" "79-0-4-4" "77-3-4-4"]
        ```
        """,
    },
    {
        "header": "Dstart and duration",
        "images": ["data/img/dstart_and_duration.png"],
    },
    # Quantization example on velocities
    {
        "header": "Velocities",
        "content": """
    ```py
    def plot_original_velocity_data(df: pd.DataFrame) -> None:
        fig, ax = plt.subplots()
        ax.scatter(df['start'], df['velocity'], color='blue')
        ax.set_xlabel('Note Start Time')
        ax.set_ylabel('Velocity')
        ax.set_title('Original Velocity Data')
        plt.show()
    ```
    """,
        "images": ["data/img/original_velocities.png"],
    },
    {
        "header": "Velocities with bin edges",
        "content": """
        ```py
        def find_dataset_velocity_bin_edges(pieces: List[ff.MidiPiece], n_bins: int = 3) -> np.ndarray:
            velocities = np.hstack([p.df.velocity.values for p in pieces])
            quantiles = np.linspace(0, 1, num=n_bins + 1)
            bin_edges = np.quantile(velocities, quantiles)
            bin_edges[0] = 0
            bin_edges[-1] = 128
            return bin_edges
        ```
        ```plaintext
        Velocity Bin Edges: [  0.  48.  60.  71.  82. 128.]
        ```
        """,
        "images": ["data/img/velocities_with_bin_edges.png"],
    },
    {
        "header": "Velocity bins",
        "content": """
            ```py
            df["velocity_bin"] = np.digitize(
                x=df['velocity'],
                bins=velocity_bin_edges,
            ) - 1
            ```
            """,
        "images": ["data/img/velocity_binned.png"],
    },
    {
        "header": "Quantization 1",
        "images": ["data/img/quantization_with_edges.png"],
    },
    {
        "header": "Quantization 2",
        "content": """
        ```py
        ["74-1-4-4", "71-0-4-4" "83-0-4-4" "79-0-4-4" "77-3-4-4"]
        ```
        """,
        "piece_paths": ["data/midi/example.mid", "data/midi/quantized_example.mid"],
    },
    {
        "header": "Initial experiments",
        "images": ["data/img/graphics2.jpg"],
    },
    # ExponentialTimeTokenizer
    {
        "header": "ExponentialTimeTokenizer",
        "code": tokens[:20],
    },
    {
        "header": "ExponentialTimeTokenizer",
        "code": tokens[:20],
        "content": exponential_time_tokens,
    },
    {
        "header": "ExponentialTimeTokenizer",
        "code": tokens[:0],
    },
    {
        "header": "ExponentialTimeTokenizer",
        "code": tokens[:1],
    },
    {
        "header": "ExponentialTimeTokenizer",
        "code": tokens[:2],
    },
    {
        "header": "ExponentialTimeTokenizer",
        "code": tokens[:3],
    },
    {
        "header": "ExponentialTimeTokenizer",
        "code": tokens[:4],
    },
    {
        "piece_paths": ["data/midi/example.mid"],
        "pieces": [untokenized_piece],
        "code": tokens[:20],
    },
    # BPE
    {
        "header": "BPE on MIDI data",
        "content": "#### Original MIDI data",
        "dataframe": piece.df,
    },
    {
        "header": "Tokenize the data",
        "content": """
        #### Original MIDI data

        #### &#8595;

        ```
        7T 6T 2T VELOCITY_12 NOTE_ON_67 7T 5T VELOCITY_16 NOTE_ON_72 6T 3T VELOCITY_16 NOTE_ON_78 ...
        ```
        """,
    },
    {
        "header": "Convert to unicode characters",
        "content": """
        #### Original MIDI data

        #### &#8595;

        ```
        7T 6T 2T VELOCITY_12 NOTE_ON_67 7T 5T VELOCITY_16 NOTE_ON_72 6T 3T VELOCITY_16 NOTE_ON_78 ...
        ```

        #### &#8595;

        #### ĻĺķğÅĝ»ķĚ¯ĻğÓķĶĢ×ĸġÝĸĤãĸķ
        """,
    },
    {
        "header": 'Split to "words"',
        "content": """
        #### Original MIDI data

        #### &#8595;

        ```
        7T 6T 2T VELOCITY_12 NOTE_ON_67 7T 5T VELOCITY_16 NOTE_ON_72 6T 3T VELOCITY_16 NOTE_ON_78 ...
        ```

        #### &#8595;

        #### ĻĺķğÅĝ»ķĚ¯ĻğÓķĶĢ×ĸġÝĸĤãĸķ

        #### &#8595;

        #### ĻĺķğÅĝ»ķĚ¯ ĻğÓķĶĢ×ĸġ ÝĸĤãĸķ

        """,
    },
    {
        "header": "Train huggingface BPE tokenizer",
        "content": """
        #### Original MIDI data

        #### &#8595;

        ```
        7T 6T 2T VELOCITY_12 NOTE_ON_67 7T 5T VELOCITY_16 NOTE_ON_72 6T 3T VELOCITY_16 NOTE_ON_78 ...
        ```

        #### &#8595;

        #### ĻĺķğÅĝ»ķĚ¯ĻğÓķĶĢ×ĸġÝĸĤãĸķ

        #### &#8595;

        #### ĻĺķğÅĝ»ķĚ¯ ĻğÓķĶĢ×ĸġ ÝĸĤãĸķ

        #### &#8595;

        #### Optimized vocabulary

        """,
    },
    # Training a GPT
    {
        "header": "Training a GPT",
        "content": """
    | Dataset                            | Train tokens | Test tokens | Validation tokens |
|------------------------------------|--------------|-------------|-------------------|
| Basic (maestro only) ExponentialTimeTokenDataset |    7,071,232 |     645,120 |         788,480   |
| Giant ExponentialTimeTokenDataset |   72,385,536 |     645,120 |         788,480   |
| Basic (maestro only) AwesomeTokensDataset     |    2,614,272 |     241,152 |         288,256   |
| Giant AwesomeTokensDataset     |   27,245,056 |     241,152 |         288,256   |
    """,
    },
    # Model generated piece example
    {
        "header": "Mozart and GPT",
        "content": """
            Model size: 302M<br>
            Tokenizer: ExponentialTimeTokenier<br>
            Input size: 512<br>
            """,
        "dual_piece": (original_piece, generated_piece),
    },
    # Generated midi pitch
    {
        "header": "Pitches of generated notes 1",
        "images": ["data/img/generated_pitch_comparison.png"],
    },
    # generated midi pitch compare
    {
        "header": "Pitches of generated notes 2",
        "images": ["data/img/generated_pitch_comparison.png", "data/img/pitch_comparison.png"],
    },
    # concatenated generations
    {
        "piece_paths": ["data/midi/d_minor_gen.mid"],
    },
    # Awesome Pitch
    {
        "header": "ExponentialTimeTokenizer vs BPE tokeizer",
        "images": ["data/img/generated_pitch_comparison.png", "data/img/awesome_pitch_comparison.png"],
    },
    # Awesome chopin
    {
        "piece_paths": ["data/midi/d_minor_gen_awesome_9.mid"],
    },
    # Future plans
    {
        "header": "Future plans",
        "images": ["data/img/piano.jpg"],
    },
    # Pianoroll
    {
        "images": ["data/img/pianoroll_webpage.png"],
    },
    # links
    {
        "content": """
    Maestro dataset: https://magenta.tensorflow.org/datasets/maestro

    Github: https://github.com/Nospoko

    My Github: https://github.com/WojciechMat

    Presentation repo: https://github.com/Nospoko/midi-pydata-london-24

    Pianoroll: https://pianoroll.io/

    Email: wmatejuk14@gmail.com
    """
    },
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
        if slide["header"] == "Augmentation":
            idx = st.number_input(label="record id", value=77)
            record = dataset[idx]
            piece = ff.MidiPiece.from_huggingface(record=record)
            st.write(piece.source)
            augmentation_review(piece=piece)

    if "code" in slide:
        st.code(slide["code"], language="python")
    if "content" in slide:
        st.write(
            slide["content"],
            unsafe_allow_html=True,
        )
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
    if "dual_piece" in slide:
        streamlit_pianoroll.from_fortepyan(piece=original_piece, secondary_piece=generated_piece)


if __name__ == "__main__":
    main()
