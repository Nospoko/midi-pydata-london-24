import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from datasets import load_dataset

from modelling.extract_notes import main as benchmark_review
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

note_columns = ["pitch", "velocity", "start", "end", "duration"]
piece.df = piece.df[note_columns]

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
        "header": "Algorithmic music composition 3",
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
    {
        "header": "Why Piano?",
        "images": ["data/img/graphics3.jpg"],
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
    {
        "header": "Scores sheet 3",
        "images": ["data/img/scores.png"],
        "piece_paths": ["data/midi/scored_piece_human.mid"],
    },
    {
        "header": "Interpreting scores",
        "images": ["data/img/note_duration_comparison0.png"],
    },
    # piano performance
    {
        "header": "Yuja Wang - Flight of a Bublebee",
        "video": "data/Yuja_Wang.mp4",
    },
    {
        "header": "Piano performance as MIDI file",
        "images": ["data/img/midi_out.jpg"],
    },
    # MIDI to DataFrame Conversion
    {
        "header": "Converting MIDI to DataFrame",
        "code": """
        import fortepyan as ff

        piece = ff.MidiPiece.from_file(path="data/midi/piano.mid")
        print(piece.df)
        """,
        "dataframe": piece.df[note_columns],
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
    {"header": "Formal vs informal vocabulary", "images": ["data/img/text_keys.png"]},
    # Modelling
    {
        "header": "LLM training pipeline",
        "content": """
        #### Overview

1. **Data Gathering** -> 2. **Tokenization** -> 3. **Training**

## Steps Involved
1. **Data Gathering**
- Examples: Web scraping, publicly available datasets, proprietary data.

2. **Tokenization**
- Convert raw text into a sequence of tokens (subwords, words, or characters).

3. **Training**
- Examples: next-token-prediction, masked language modelling.
        """,
    },
    {
        "header": "MIDI LLM training pipeline",
        "content": """
        #### Overview

1. **Data Gathering** -> 2. **Tokenization** -> 3. **Training**

## Steps Involved

1. **Data Gathering**
- Examples: Crowd-sourcing, buying recordings.

2. **Tokenization**
- Convert MIDI data into a sequence of tokens (many methods to use!).

3. **Pre-Training**
- Examples: next-token-prediction, masked music modelling.


        """,
    },
    {
        "header": "Initial experiments",
        "images": ["data/img/graphics2.jpg"],
    },
    {
        "header": "Initial experiments",
        "content": """
        #### Initial Plan
        - LLM for Seq-to-Seq
        """,
    },
    {
        "header": "Initial experiments",
        "content": """
        #### Initial Plan
        - LLM for Seq-to-Seq

        #### Experiments
        - Diffusion Models
        - VQ-VAE
        - LLM for Note Pitches
        """,
    },
    {
        "header": "Initial experiments",
        "content": """
        #### Initial Plan
        - LLM for Seq-to-Seq

        #### Experiments
        - Diffusion Models
        - VQ-VAE
        - LLM for Note Pitches

        #### Final Experiment
        - GPT for Seq-to-Seq
        """,
    },
    {
        "header": "Modelling piano performances with Large Language Models",
        "images": ["data/img/piano.jpg"],
    },
    {"header": "Dataset sizes comparison", "images": ["data/img/training_dataset_sizes.png"]},
    # Augmentation
    {
        "header": "Augmentation 1",
        "content": """
            #### Pitch shifting
            ```py
            def pitch_shift(df: pd.DataFrame, shift: int = 1) -> pd.DataFrame:
                df.pitch += shift
                return df, shift
            ```
            """,
    },
    {
        "header": "Augmentation 2",
        "content": """
            #### Pitch shifting
            ```py
            def pitch_shift(df: pd.DataFrame, shift: int = 1) -> pd.DataFrame:
                df.pitch += shift
                return df, shift
            ```
            """,
        "images": ["data/img/pitch_shifted.png"],
    },
    {
        "header": "Augmentation 3",
        "content": """
            #### Speed change
            ```py
            def change_speed(df: pd.DataFrame, factor: float = None) -> pd.DataFrame:
                df.start /= factor
                df.end /= factor
                df.duration = df.end - df.start
                return df
            """,
    },
    {
        "header": "Augmentation 4",
        "content": """
            #### Speed change
            ```py
            def change_speed(df: pd.DataFrame, factor: float = None) -> pd.DataFrame:
                df.start /= factor
                df.end /= factor
                df.duration = df.end - df.start
                return df

            piece = ff.MidiPiece.from_file("data/midi/d_minor_bach.mid")
            change_speed(df=piece.df, factor=1.05)
            """,
        "piece_paths": ["data/midi/d_minor_bach.mid", "data/midi/d_minor_bach_speeded.mid"],
    },
    # Quantization
    {
        "header": "Quantization 1",
        "content": """
    ```py
    def quantize_series(series, step_size) -> np.ndarray:
        # Round the series to the nearest step size
        quantized = np.round(series / step_size) * step_size
        return quantized
    ```
    """,
        "dataframe": pd.read_csv("data/quantization.csv", index_col=0),
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
        "code": tokens[:20],
        "content": """
            #### Original Data:
            | pitch | velocity |   start   |     end    |
            |-------|----------|-----------|------------|
            |   59  |    94    |  0.000000 |  0.072727  |
            |   48  |    77    |  0.077273 |  0.177273  |
            |   60  |    95    |  0.102273 |  0.229545  |
            |   47  |    79    |  0.159091 |  0.275000  |
            #### Split to events:
            | pitch | velocity |      time | event_type |
            |-------|----------|-----------|------------|
            |    59 |       94 |  0.000000 |    note_on |
            |    59 |       94 |  0.072727 |   note_off |
            |    48 |       77 |  0.077273 |    note_on |
            |    60 |       95 |  0.102273 |    note_on |
            |    47 |       79 |  0.159091 |    note_on |
            |    48 |       77 |  0.177273 |   note_off |
            |    60 |       95 |  0.229545 |   note_off |
            |    47 |       79 |  0.275000 |   note_off |

            """,
    },
    {
        "header": "ExponentialTimeTokenizer",
        "code": tokens[:1],
        "content": """
            | pitch | velocity |      time | event_type |
            |-------|----------|-----------|------------|
            |    59 |       <span style="color:red">94</span> |  0.000000 |    note_on |
            |    59 |       94 |  0.072727 |   note_off |
            |    48 |       77 |  0.077273 |    note_on |
            |    60 |       95 |  0.102273 |    note_on |
            |    47 |       79 |  0.159091 |    note_on |
            |    48 |       77 |  0.177273 |   note_off |
            |    60 |       95 |  0.229545 |   note_off |
            |    47 |       79 |  0.275000 |   note_off |

            """,
    },
    {
        "header": "ExponentialTimeTokenizer",
        "code": tokens[:2],
        "content": """
            | pitch | velocity |      time | event_type |
            |-------|----------|-----------|------------|
            |    <span style="color:red">59</span> |       94 |  0.000000 |    <span style="color:red">note_on |
            |    59 |       94 |  0.072727 |   note_off |
            |    48 |       77 |  0.077273 |    note_on |
            |    60 |       95 |  0.102273 |    note_on |
            |    47 |       79 |  0.159091 |    note_on |
            |    48 |       77 |  0.177273 |   note_off |
            |    60 |       95 |  0.229545 |   note_off |
            |    47 |       79 |  0.275000 |   note_off |

            """,
    },
    {
        "header": "ExponentialTimeTokenizer",
        "code": tokens[:3],
        "content": """
            | pitch | velocity |      time | event_type |
            |-------|----------|-----------|------------|
            |    59 |       94 |  <span style="color:red">0.000000 |    note_on |
            |    59 |       94 |  <span style="color:red">0.072727</span> |   note_off |
            |    48 |       77 |  0.077273 |    note_on |
            |    60 |       95 |  0.102273 |    note_on |
            |    47 |       79 |  0.159091 |    note_on |
            |    48 |       77 |  0.177273 |   note_off |
            |    60 |       95 |  0.229545 |   note_off |
            |    47 |       79 |  0.275000 |   note_off |

            """,
    },
    {
        "header": "ExponentialTimeTokenizer",
        "code": tokens[:5],
        "content": """
            | pitch | velocity |      time | event_type |
            |-------|----------|-----------|------------|
            |    59 |       94 |  0.000000 |    note_on |
            | <span style="color:red">59 | <span style="color:red">94 |  0.072727 | <span style="color:red">note_off |
            |    48 |       77 |  0.077273 |    note_on |
            |    60 |       95 |  0.102273 |    note_on |
            |    47 |       79 |  0.159091 |    note_on |
            |    48 |       77 |  0.177273 |   note_off |
            |    60 |       95 |  0.229545 |   note_off |
            |    47 |       79 |  0.275000 |   note_off |

            """,
    },
    {
        "header": "ExponentialTimeTokenizer",
        "code": tokens[:5],
        "content": """
            | pitch | velocity |      time | event_type |
            |-------|----------|-----------|------------|
            |    59 |       94 |  0.000000 |    note_on |
            |    59 |       94 |  <span style="color:red">0.072727 |   note_off |
            |    48 |       77 |  <span style="color:red">0.077273 |    note_on |
            |    60 |       95 |  0.102273 |    note_on |
            |    47 |       79 |  0.159091 |    note_on |
            |    48 |       77 |  0.177273 |   note_off |
            |    60 |       95 |  0.229545 |   note_off |
            |    47 |       79 |  0.275000 |   note_off |

            """,
    },
    {
        "header": "ExponentialTimeTokenizer",
        "code": tokens[:7],
        "content": """
            | pitch | velocity |      time | event_type |
            |-------|----------|-----------|------------|
            |    59 |       94 |  0.000000 |    note_on |
            |    59 |       94 |  0.072727 |   note_off |
            | <span style="color:red">48 | <span style="color:red">77 |  0.077273 | <span style="color:red">note_on |
            |    60 |       95 |  0.102273 |    note_on |
            |    47 |       79 |  0.159091 |    note_on |
            |    48 |       77 |  0.177273 |   note_off |
            |    60 |       95 |  0.229545 |   note_off |
            |    47 |       79 |  0.275000 |   note_off |

            """,
    },
    {
        "header": "ExponentialTimeTokenizer",
        "code": tokens[:8],
        "content": """
            | pitch | velocity |      time | event_type |
            |-------|----------|-----------|------------|
            |    59 |       94 |  0.000000 |    note_on |
            |    59 |       94 |  0.072727 |   note_off |
            |    48 |       77 |  <span style="color:red">0.077273 |    note_on |
            |    60 |       95 |  <span style="color:red">0.102273 |    note_on |
            |    47 |       79 |  0.159091 |    note_on |
            |    48 |       77 |  0.177273 |   note_off |
            |    60 |       95 |  0.229545 |   note_off |
            |    47 |       79 |  0.275000 |   note_off |

            """,
    },
    {
        "header": "ExponentialTimeTokenizer",
        "code": tokens[:10],
        "content": """
            | pitch | velocity |      time | event_type |
            |-------|----------|-----------|------------|
            |    59 |       94 |  0.000000 |    note_on |
            |    59 |       94 |  0.072727 |   note_off |
            |    48 |       77 |  0.077273 |    note_on |
            | <span style="color:red">60 | <span style="color:red">95 |  0.102273 | <span style="color:red">note_on |
            |    47 |       79 |  0.159091 |    note_on |
            |    48 |       77 |  0.177273 |   note_off |
            |    60 |       95 |  0.229545 |   note_off |
            |    47 |       79 |  0.275000 |   note_off |

            """,
    },
    {
        "piece_paths": ["data/midi/example.mid"],
        "pieces": [untokenized_piece],
        "code": tokens[:20],
    },
    # BPE
    {
        "header": "Tokenization in NLP",
        "images": ["data/img/tokenization_nlp.png"],
    },
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

        #### ["ĻĺķğÅĝ»ķĚ¯ĻğÓķĶĢ×ĸġÝĸĤãĸķ"]

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

        #### ["ĻĺķğÅĝ»ķĚ¯ĻğÓķĶĢ×ĸġÝĸĤãĸķ"]

        #### &#8595;

        #### ["ĻĺķğÅĝ»ķĚ¯", "ĻğÓķĶĢ×ĸġ", "ÝĸĤãĸķ"]

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

        #### ["ĻĺķğÅĝ»ķĚ¯ĻğÓķĶĢ×ĸġÝĸĤãĸķ"]

        #### &#8595;

        #### ["ĻĺķğÅĝ»ķĚ¯", "ĻğÓķĶĢ×ĸġ", "ÝĸĤãĸķ"]

        #### &#8595;

        | Index | Token     |
        |-------|-----------|
        | 0     | ĹķĶğ      |
        | 1     | Åĝ»ķ      |
        | 2     | Ĺğ        |
        | 3     | §         |
        | 4     | ĹĶĝ       |
        | 5     | ĶĢ×       |
        | 6     | ĸķĶğ      |
        | 7     | ġ     |
        | 8     | ĸķğ       |
        | 9     | ¨ġ§       |

        """,
    },
    # Training a GPT
    {
        "header": "Training a GPT",
        "content": """
| Dataset                                   | Train tokens | Test tokens | Validation tokens |
|-------------------------------------------|--------------|-------------|-------------------|
| Basic (maestro only) ExponentialTimeTokenDataset | 7,071,232    | 645,120     | 788,480           |
| Giant ExponentialTimeTokenDataset         | 72,385,536   | 645,120     | 788,480           |
| Colossal ExponentialTimeTokenDataset      | 210,522,112  | 645,120     | 788,480           |
| Basic (maestro only) AwesomeTokensDataset    | 2,614,272    | 241,152     | 288,256           |
| Giant AwesomeTokensDataset                   | 27,245,056   | 241,152     | 288,256           |
| Colossal AwesomeTokensDataset                | 77,072,896   | 242,176     | 288,768           |
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
    # Future plans
    {
        "header": "Future plans",
        "images": ["data/img/piano.jpg"],
    },
    {"header": "Benchmark task example"},
    {
        "header": "Open Problems in Modelling Piano Performances",
        "content": """
            #### Effective Metrics
            - Defining metrics to track training progress and compare models.
            - Algorithmically determining if a musical piece sounds good and possesses the right structure.
            - Examples:
                - Assessing tempo consistency (approximating beats per minute).
                - Checking if pitches are from the same key as the rest of the piece.

            #### Algorithmic Music Generation
            - Training models on algorithmically generated music.
            - Combining real music performances with algorithmic composition.
            - Exploring and discovering algorithms for creating music.

            #### Call to Action
            - https://pianoroll.io
            """,
    },
    # Pianoroll
    {
        "images": ["data/img/pianoroll_webpage.png"],
    },
    {
        "header": "Conclusions",
        "content": """
            #### The Exciting World of Piano Performance Music
            - Piano music - the universe of artistry and mathematical relationships.
            - Transformer-based architectures might be just what we need.

            #### Building Communities and Sharing Knowledge
            - pianoroll.io

            #### Future Prospects
            - The future of algorithmic music seems bright.
            - By focusing more resources and attention on this problem, we might create something truly remarkable.
            - Humans have been discovering and understanding the mathematical nature of music for ages.

            """,
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
    col1, col2, col3 = st.columns([1, 5, 1])
    if col1.button("Previous"):
        if st.session_state.slide > 0:
            st.session_state.slide -= 1
    if col3.button("Next"):
        if st.session_state.slide < len(slides) - 1:
            st.session_state.slide += 1
    pres_columns = st.columns([1, 5, 1])
    with pres_columns[1]:
        # Display the current slide
        slide = slides[st.session_state.slide]

        if "header" in slide:
            st.header(slide["header"])
            # Make Visualisation slide responsive
            if slide["header"] == "Visualising and Listening to MIDI Files":
                st.code(slide["code"], language="python")
                idx = st.number_input(label="record id", value=77)
                record = dataset[idx]
                piece = ff.MidiPiece.from_huggingface(record=record)
                st.json(piece.source, expanded=False)
                streamlit_pianoroll.from_fortepyan(piece=piece)
                return
            if slide["header"] == "Benchmark task example":
                benchmark_review()

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
