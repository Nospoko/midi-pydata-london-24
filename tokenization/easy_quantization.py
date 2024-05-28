import numpy as np
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll


# Function to quantize values by rounding to the nearest step size
def quantize_series(series: np.ndarray, step_size: float) -> np.ndarray:
    # Round the series to the nearest step size
    quantized = np.round(series / step_size) * step_size
    return quantized


def main():
    # Load the MIDI file
    piece = ff.MidiPiece.from_file("data/midi/d_minor_bach.mid")

    # Calculate dstart
    next_start = piece.df.start.shift(-1)
    piece.df["dstart"] = (next_start - piece.df.start).fillna(0)

    # Define the granularity (step size) for each series
    dstart_step = 0.05
    duration_step = 0.05
    velocity_step = 5

    # Apply quantization to dstart, duration, and velocity
    piece.df["velocity_quantized"] = quantize_series(piece.df.velocity, velocity_step)
    piece.df["duration_quantized"] = quantize_series(piece.df.duration, duration_step)
    piece.df["dstart_quantized"] = quantize_series(piece.df.dstart, dstart_step)

    # Print an example of the quantized DataFrame
    st.write(piece.df)
    quantized = piece.df.copy()
    quantized["velocity"] = quantized["velocity_quantized"]
    quantized["duration"] = quantized["duration_quantized"]
    quantized["dstart"] = quantized["dstart_quantized"]

    quantized["start"] = quantized["dstart"].cumsum()
    quantized["end"] = quantized["start"] + quantized["duration"]

    streamlit_pianoroll.from_fortepyan(ff.MidiPiece(quantized))


if __name__ == "__main__":
    main()
