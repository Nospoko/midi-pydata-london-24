import fortepyan as ff
import streamlit as st
import streamlit_pianoroll

from browse_dataset import select_record, select_dataset


def main():
    dataset = select_dataset()
    record = select_record(midi_dataset=dataset)

    piece = ff.MidiPiece.from_huggingface(record=record)
    notes_no_bass = piece.df[piece.df.pitch > 48]
    notes_bass = piece.df[piece.df.pitch <= 48]

    piece_no_bass = ff.MidiPiece(notes_no_bass)
    piece_bass = ff.MidiPiece(notes_bass)

    st.write("Piece without bass")
    streamlit_pianoroll.from_fortepyan(piece_no_bass)
    st.write("Only bass")
    streamlit_pianoroll.from_fortepyan(piece_bass)
    st.write("Together")
    streamlit_pianoroll.from_fortepyan(piece_no_bass, secondary_piece=piece_bass)


if __name__ == "__main__":
    main()
