import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll

from browse_dataset import select_record, select_dataset


def change_speed(df: pd.DataFrame, factor: float = None) -> pd.DataFrame:
    df.start /= factor
    df.end /= factor
    df.duration = df.end - df.start
    return df


def pitch_shift(df: pd.DataFrame, shift: int = 5) -> pd.DataFrame:
    # No more than given number of steps
    PITCH_LOW = 21
    PITCH_HI = 108
    low_shift = -min(shift, df.pitch.min() - PITCH_LOW)
    high_shift = min(shift, PITCH_HI - df.pitch.max())

    if low_shift > high_shift:
        shift = 0
        print("Pitch shift edge case:", df.pitch.min(), df.pitch.max())
    df.pitch += shift
    return df


def augment_piece(piece: ff.MidiPiece, speed_factor: float, shift: int) -> ff.MidiPiece:
    speed_change_df = change_speed(
        piece.df.copy(),
        factor=speed_factor,
    )
    pitch_shift_df = pitch_shift(
        speed_change_df,
        shift=shift,
    )
    source = piece.source | {
        "shift": shift,
        "factor": speed_factor,
    }
    augmented_piece = ff.MidiPiece(pitch_shift_df, source=source)
    return augmented_piece


def augmentation_review(piece: ff.MidiPiece):
    shift = st.number_input(label="pitch shift", value=5)
    speed_factor = st.number_input(label="speed factor", value=1.0)

    augmented_piece = augment_piece(
        piece=piece,
        speed_factor=speed_factor,
        shift=shift,
    )

    comparison_columns = st.columns(2)
    with comparison_columns[0]:
        streamlit_pianoroll.from_fortepyan(piece=piece)
    with comparison_columns[1]:
        streamlit_pianoroll.from_fortepyan(piece=augmented_piece)


def main():
    dataset = select_dataset()
    record = select_record(midi_dataset=dataset)
    piece = ff.MidiPiece.from_huggingface(record=record)

    augmentation_review(piece=piece)


if __name__ == "__main__":
    main()
