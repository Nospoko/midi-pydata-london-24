import fortepyan as ff


def main():
    piece = ff.MidiPiece.from_file(path="data/midi/piano.mid")
    print(piece.df)


if __name__ == "__main__":
    main()
