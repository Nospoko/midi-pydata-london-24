import pandas as pd
import fortepyan as ff


def main():
    notes = []
    velocity = 127
    time = 0
    dt = 0.1
    for pitch in range(21, 109):
        notes.append(
            {
                "pitch": pitch,
                "start": time,
                "end": time + dt,
                "velocity": velocity,
            }
        )
        velocity -= 1
        time += dt
    for pitch in range(0, 89):
        notes.append(
            {
                "pitch": 109 - pitch,
                "start": time,
                "end": time + dt,
                "velocity": velocity,
            }
        )
        velocity += 1
        time += dt
    notes = pd.DataFrame(notes)
    piece = ff.MidiPiece(df=notes)
    piece_path = "data/midi/generated_midi.mid"
    piece.to_midi().write(piece_path)


if __name__ == "__main__":
    main()
