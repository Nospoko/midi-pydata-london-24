import librosa
import numpy as np
import fortepyan as ff
import matplotlib.pyplot as plt


def plot_spectrogram():
    y, sr = librosa.load("data/piano.mp3", sr=44100)
    n_fft = 2 << 15
    hop_length = n_fft // 4

    spectrogram = librosa.stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    S_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)

    fig, ax = plt.subplots()
    librosa.display.specshow(
        data=S_db,
        n_fft=n_fft,
        hop_length=hop_length,
        y_axis="fft_note",
        x_axis="s",
        sr=sr,
        ax=ax,
    )

    # Limit axes to the frequencies of a piano
    f_max = librosa.note_to_hz("C8")
    f_min = librosa.note_to_hz("A0")
    ax.set_ylim((f_min, f_max))
    return fig


def plot_pianoroll():
    piece = ff.MidiPiece.from_file(path="data/piano.mid")
    fig = ff.view.draw_pianoroll_with_velocities(midi_piece=piece)
    return fig


if __name__ == "__main__":
    spec = plot_spectrogram()
    pianoroll = plot_pianoroll()
    spec.savefig("data/img/spectrogram.png")
    pianoroll.savefig("data/img/pianoroll.png")
