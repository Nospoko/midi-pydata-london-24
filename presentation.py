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
def prepare_piece(idx: int = 78):
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

quantized_df_columns = [
    "pitch",
    "velocity",
    "velocity_quantized",
    "start",
    "start_quantized",
    "end",
    "end_quantized",
]

# Initialize session state
if "slide" not in st.session_state:
    st.session_state.slide = 0

note_columns = ["pitch", "velocity", "start", "end"]
piece.df = piece.df[note_columns]
two_notes = []
two_notes.append(
    {
        "pitch": 43,
        "start": 0.4,
        "end": 2.5,
        "velocity": 67,
    }
)
two_notes.append(
    {
        "pitch": 48,
        "start": 0.6,
        "end": 2.8,
        "velocity": 98,
    }
)
two_notes = pd.DataFrame(two_notes)

# Slide content
slides = [
    # Opening slide
    {
        "images": ["data/img/cover.png"],
    },
    # Pythagoreans
    {
        "header": '"There is geometry in the humming of the strings, and there is music in the spacing of the spheres"',
        "images": ["data/img/pythagoras.png"],
    },
    {
        "header": "Algorithmic music composition",
        "images": {"data/img/mozart_table.png"},
    },
    # Hiller
    {
        "header": "The Illiac Suite",
        "content": """
        - ILLIAC computer weight about 1.8 TONS
        - had a total of 64 kb of memory
        """,
        "images": ["data/img/hiller.jpg"],
    },
    # Cage
    {
        "header": "HPSCHD",
        "images": ["data/img/cage.jpg"],
    },
    # Piano
    {
        "header": "Piano",
        "images": ["data/img/graphics3.jpg"],
    },
    {
        "header": "Key-press schema",
        "images": ["data/img/piano_diagram.png"],
    },
    # pedals
    {
        "header": "Disclaimer: pedals!",
        "images": ["data/img/pedals.jpg"],
    },
    {
        "header": "MIDI - Musical Instrument Digital Interface",
        "images": ["data/img/midi_out.jpg"],
    },
    {
        "header": "MIDI - Musical Instrument Digital Interface",
        "images": ["data/img/key_press.png"],
        "dataframe": two_notes,
    },
    # piano performance
    {
        "header": "Yuja Wang - Flight of a Bublebee",
        "video": "data/Yuja_Wang.mp4",
    },
    {
        "header": "MIDI - Musical Instrument Digital Interface",
        "images": ["data/img/pianoroll_example.png"],
        "dataframe": piece.df,
    },
    {"header": "Audio vs MIDI", "images": ["data/img/spectrogram.png"]},
    {"header": "Audio vs MIDI", "images": ["data/img/pianoroll.png"], "piece_paths": ["data/midi/d_minor_bach.mid"]},
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
    # Scores
    {
        "header": "Scores sheet",
        "images": ["data/img/scores.png"],
    },
    # Scored piece
    {
        "header": "Mechanical vs human",
        "piece_paths": ["data/midi/scored_piece.mid", "data/midi/scored_piece_human.mid"],
    },
    {
        "header": "Interpreting scores",
        "images": ["data/img/note_duration_comparison0.png", "data/img/dstart_comparison.png"],
    },
    {
        "header": "Maestro",
        "content": """
- https://magenta.tensorflow.org/datasets/maestro

- High quality dataset with classical piano music.

- Backbone of our datasets.


| Split       | Performances | Duration (hours) | Size (GB) | Notes (millions) |
|-------------|--------------|------------------|-----------|------------------|
| Train       | 962          | 159.2            | 96.3      | 5.66             |
| Validation  | 137          | 19.4             | 11.8      | 0.64             |
| Test        | 177          | 20.0             | 12.1      | 0.74             |
| **Total**   | 1276         | 198.7            | 120.2     | 7.04             |

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
    # pitch
    {
        "header": "Pitch distribution",
        "images": ["data/img/pitch_comparison.png"],
    },
    {"header": "Formal vs informal vocabulary", "images": ["data/img/text_keys.png"]},
    # Modelling
    {
        "header": "MIDI LLM training pipeline",
        "content": """
        #### Overview

1. **Data Gathering** -> 2. **Tokenization** -> 3. **Training**

## Steps Involved

1. **Data Gathering**
- Examples: Web scraping, publicly available datasets, proprietary data


2. **Tokenization**
- Convert raw data into a sequence of tokens (many methods to use!).

3. **Training**
- Examples: next-token-prediction, masked sequence modelling.


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
        "video": "data/chopin-a-minor.mp4",
        "images": ["data/img/diffused_cat.png"],
    },
    {
        "header": "Don't be a hero",
        "content": """
        ~Andrej Karpathy
        """,
    },
    {
        "header": "Modelling piano performances with Large Language Models",
        "images": ["data/img/piano.jpg"],
    },
    {"header": "Dataset sizes comparison", "images": ["data/img/training_dataset_sizes.png"]},
    # Augmentation
    {
        "header": "Augmentation",
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
        "header": "Augmentation",
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
        "header": "Augmentation",
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
        "header": "Augmentation",
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
    {
        "header": "Tokenization in NLP",
        "images": ["data/img/tokenization_nlp.png"],
    },
    # Quantization
    {
        "header": "Normalisation - Quantization",
        "content": """
    ```py
    def quantize_series(series, step_size) -> np.ndarray:
        # Round the series to the nearest step size
        quantized = np.round(series / step_size) * step_size
        return quantized
    ```
    """,
        "dataframe": pd.read_csv("data/quantization.csv", index_col=0)[quantized_df_columns],
    },
    # ExponentialTimeTokenizer
    {
        "header": "Pre-tokenization: Transcription",
        "code": tokens[:20],
    },
    {
        "header": "Pre-tokenization: Transcription",
        "code": tokens[:20],
        "content": exponential_time_tokens,
    },
    {
        "header": "Pre-tokenization: Transcription",
        "code": tokens[:20],
        "content": """
#### Original Data:
| pitch | velocity |   start   |     end    |
|-------|----------|-----------|------------|
|   59  |    94    |  0.000000 |  0.072727  |
|   48  |    77    |  0.077273 |  0.177273  |
|   60  |    95    |  0.102273 |  0.229545  |
|   47  |    79    |  0.159091 |  0.275000  |
#### Events:
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
        "header": "Transcription process",
        "code": tokens[:1],
        "content": """
#### Original Data:
| pitch | velocity |   start   |     end    |
|-------|----------|-----------|------------|
|   59  |    <span style="color:red">94    |  0.000000 |  0.072727  |
|   48  |    77    |  0.077273 |  0.177273  |
|   60  |    95    |  0.102273 |  0.229545  |
|   47  |    79    |  0.159091 |  0.275000  |
#### Events:
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
        "images": ["data/img/tokenization12.png"],
    },
    {
        "header": "Transcription process",
        "code": tokens[:2],
        "content": """
#### Original Data:
| pitch | velocity |   start   |     end    |
|-------|----------|-----------|------------|
|   <span style="color:red">59  |    94    |  <span style="color:red">0.000000 |  0.072727  |
|   48  |    77    |  0.077273 |  0.177273  |
|   60  |    95    |  0.102273 |  0.229545  |
|   47  |    79    |  0.159091 |  0.275000  |
#### Events:
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
        "images": ["data/img/tokenization12.png"],
    },
    {
        "header": "Transcription process",
        "code": tokens[:3],
        "content": """
#### Original Data:
| pitch | velocity |   start   |     end    |
|-------|----------|-----------|------------|
|   59  |    94    |  <span style="color:red">0.000000 |  <span style="color:red">0.072727  |
|   48  |    77    |  0.077273 |  0.177273  |
|   60  |    95    |  0.102273 |  0.229545  |
|   47  |    79    |  0.159091 |  0.275000  |
#### Events:
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
        "images": ["data/img/tokenization3.png"],
    },
    {
        "header": "Transcription process",
        "code": tokens[:5],
        "content": """
            #### Original Data:
| pitch | velocity |   start   |     end    |
|-------|----------|-----------|------------|
|   <span style="color:red">59  |    <span style="color:red">94    |  0.000000 |  <span style="color:red">0.072727  |
|   48  |    77    |  0.077273 |  0.177273  |
|   60  |    95    |  0.102273 |  0.229545  |
|   47  |    79    |  0.159091 |  0.275000  |
#### Events:
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
        "images": ["data/img/tokenization45.png"],
    },
    {
        "header": "Transcription process",
        "code": tokens[:5],
        "content": """
#### Original Data:
| pitch | velocity |   start   |     end    |
|-------|----------|-----------|------------|
|   59  |    94    |  0.000000 |  <span style="color:red">0.072727  |
|   48  |    77    |  <span style="color:red">0.077273 |  0.177273  |
|   60  |    95    |  0.102273 |  0.229545  |
|   47  |    79    |  0.159091 |  0.275000  |
#### Events:
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
        "images": ["data/img/tokenization45.png"],
    },
    {
        "header": "Transcription process",
        "code": tokens[:7],
        "content": """
#### Original Data:
| pitch | velocity |   start   |     end    |
|-------|----------|-----------|------------|
|   59  |    94    |  0.000000 |  0.072727  |
|   <span style="color:red">48  |    <span style="color:red">77    |  <span style="color:red">0.077273 |  0.177273  |
|   60  |    95    |  0.102273 |  0.229545  |
|   47  |    79    |  0.159091 |  0.275000  |
#### Events:
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
        "images": ["data/img/tokenization6.png"],
    },
    {
        "header": "Transcription process",
        "code": tokens[:8],
        "content": """
#### Original Data:
| pitch | velocity |   start   |     end    |
|-------|----------|-----------|------------|
|   59  |    94    |  0.000000 |  0.072727  |
|   48  |    77    |  <span style="color:red">0.077273 |  0.177273  |
|   60  |    95    |  <span style="color:red">0.102273 |  0.229545  |
|   47  |    79    |  0.159091 |  0.275000  |
#### Events:
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
        "images": ["data/img/tokenization7.png"],
    },
    {
        "header": "Transcription process",
        "code": tokens[:10],
        "content": """
#### Original Data:
| pitch | velocity |   start   |     end    |
|-------|----------|-----------|------------|
|   59  |    94    |  0.000000 |  0.072727  |
|   48  |    77    |  0.077273 |  0.177273  |
|   <span style="color:red">60  |    <span style="color:red">95    |  <span style="color:red">0.102273 |  0.229545  |
|   47  |    79    |  0.159091 |  0.275000  |
#### Events:
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
        "images": ["data/img/tokenization8.png"],
    },
    {
        "piece_paths": ["data/midi/example.mid"],
        "pieces": [untokenized_piece],
        "code": tokens[:20],
    },
    {
        "header": "BPE tokenizer",
        "images": ["data/img/hello_pydata.png"],
    },
    {
        "header": "BPE on MIDI",
        "images": ["data/img/bpe.png"],
    },
    {
        "header": "Vocabulary example",
        "content": """
| Index | Merged Tokens                           | n_symbols |
|-------|-----------------------------------------|-----------|
| 0     | ["NOTE_OFF_72", "3T", "VELOCITY_12"]    | 3         |
| 1     | ["NOTE_OFF_76", "3T", "2T", "VELOCITY_14"] | 4         |
| 2     | ["NOTE_OFF_69", "3T", "VELOCITY_17"]    | 3         |
| 3     | ["NOTE_OFF_79", "VELOCITY_21"]          | 2         |
| 4     | ["NOTE_OFF_35", "VELOCITY_10"]          | 2         |
| 5     | ["NOTE_ON_63", "4T", "2T", "VELOCITY_16"] | 4         |
| 6     | ["NOTE_OFF_92", "VELOCITY_10"]          | 2         |
| 7     | ["NOTE_OFF_55", "3T", "VELOCITY_18"]    | 3         |
| 8     | ["NOTE_OFF_75", "3T", "2T", "VELOCITY_12"] | 4         |
| 9     | ["NOTE_ON_51", "4T", "2T", "VELOCITY_17"] | 4         |
| 10    | ["NOTE_ON_50", "4T"]                    | 2         |
| 11    | ["NOTE_ON_43", "4T", "2T", "VELOCITY_13"] | 4         |

        """,
    },
    # Training a GPT
    {
        "header": "Training a GPT",
        #         "content": """
        # | Dataset                                   | Train tokens | Test tokens | Validation tokens |
        # |-------------------------------------------|--------------|-------------|-------------------|
        # | Basic (maestro only) ExponentialTimeTokenDataset | 7,071,232    | 645,120     | 788,480           |
        # | Giant ExponentialTimeTokenDataset         | 72,385,536   | 645,120     | 788,480           |
        # | Colossal ExponentialTimeTokenDataset      | 210,522,112  | 645,120     | 788,480           |
        # | Basic (maestro only) AwesomeTokensDataset    | 2,614,272    | 241,152     | 288,256           |
        # | Giant AwesomeTokensDataset                   | 27,245,056   | 241,152     | 288,256           |
        # | Colossal AwesomeTokensDataset                | 77,072,896   | 242,176     | 288,768           |
        #     """,
        "images": ["data/img/wandb.png"],
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
        "header": "Pitches of generated notes",
        "images": ["data/img/generated_pitch_comparison.png"],
    },
    # generated midi pitch compare
    {
        "header": "Pitches of generated notes",
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
    {
        "header": "Future plans",
        "content": """
            - Defining benchmark tasks
            - Building a community
            - Sharing knowledge
            - Experimenting more!
            """,
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
    # Pianoroll
    {
        "images": ["data/img/pianoroll_page.png"],
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
    {
        "header": "Thank you very much!",
        "images": ["data/img/graphics4.jpg"],
    },
    # links
    {
        "header": "Thank you very much!",
        "content": """
    Maestro dataset: https://magenta.tensorflow.org/datasets/maestro

    Github: https://github.com/Nospoko

    My Github: https://github.com/WojciechMat

    Presentation repo: https://github.com/Nospoko/midi-pydata-london-24

    Pianoroll: https://pianoroll.io/

    Email: wmatejuk14@gmail.com
    """,
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

            if slide["header"] == "Mechanical vs human":
                # side by side comparison
                display_columns = st.columns([1, 1])
                with display_columns[0]:
                    prepared_piece = ff.MidiPiece.from_file(slide["piece_paths"][0])
                    streamlit_pianoroll.from_fortepyan(piece=prepared_piece)
                with display_columns[1]:
                    prepared_piece = ff.MidiPiece.from_file(slide["piece_paths"][1])
                    streamlit_pianoroll.from_fortepyan(piece=prepared_piece)
                return

            # custom tokenization slides
            if slide["header"] == "Transcription process":
                st.code(slide["code"], language="python")
                display_columns = st.columns([1, 1, 3])
                with display_columns[0]:
                    st.write(slide["content"], unsafe_allow_html=True)
                with display_columns[2]:
                    st.image(slide["images"][0])
                return

        if "code" in slide:
            st.code(slide["code"], language="python")
        if "content" in slide:
            st.write(
                slide["content"],
                unsafe_allow_html=True,
            )
        if "images" in slide and "dataframe" in slide:
            display_columns = st.columns([2, 5])
            with display_columns[0]:
                st.dataframe(slide["dataframe"])
            with display_columns[1]:
                st.image(slide["images"][0])
            return
        if "images" in slide:
            for image in slide["images"]:
                st.image(image=image)
        if "dataframe" in slide:
            st.write(slide["dataframe"])
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
