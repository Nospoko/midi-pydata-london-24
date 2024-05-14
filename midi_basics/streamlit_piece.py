import fortepyan as ff
import streamlit_pianoroll
from datasets import load_dataset


def main():
    dataset = load_dataset("roszcz/maestro-sustain-v2", split="test")
    record = dataset[77]

    piece = ff.MidiPiece.from_huggingface(record=record)
    streamlit_pianoroll.from_fortepyan(piece=piece)


if __name__ == "__main__":
    main()
