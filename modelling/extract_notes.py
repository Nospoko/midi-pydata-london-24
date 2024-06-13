import fortepyan as ff
import streamlit as st
import streamlit_pianoroll

from browse_dataset import select_record, select_dataset


def main():
    dataset = select_dataset()
    record = select_record(midi_dataset=dataset)

    piece = ff.MidiPiece.from_huggingface(record=record)
    high = st.number_input(label="high pitch", value=60)
    low = st.number_input(label="low pitch", value=0)
    note_ids = (piece.df.pitch <= high) & (piece.df.pitch >= low)
    notes_deprived = piece.df[~note_ids]
    notes_extracted = piece.df[note_ids]

    piece_deprived = ff.MidiPiece(notes_deprived)
    piece_extracted = ff.MidiPiece(notes_extracted)

    st.write("Together")
    streamlit_pianoroll.from_fortepyan(piece_deprived, secondary_piece=piece_extracted)
    st.write(f"Piece without notes from range {low} - {high}")
    streamlit_pianoroll.from_fortepyan(piece_deprived)
    st.write("Extracted notes")
    streamlit_pianoroll.from_fortepyan(piece_extracted)


if __name__ == "__main__":
    main()
