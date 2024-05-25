import librosa
import numpy as np
import fortepyan as ff
import matplotlib.pyplot as plt
from fortepyan.view.pianoroll.structures import FigureResolution


def plot_spectrogram():
    y, sr = librosa.load("data/generated_audio.mp3", sr=44100)
    n_fft = 2 << 15
    hop_length = n_fft // 4

    spectrogram = librosa.stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    S_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
    # Plotting
    width_px = 1920
    height_px = 1080

    dpi = 120

    fig_width = width_px / dpi
    fig_height = height_px / dpi
    figsize = (fig_width, fig_height)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
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
    piece = ff.MidiPiece.from_file(path="data/midi/generated_midi.mid")[53:58]
    figres = FigureResolution(1920, 1080, dpi=120)
    fig = ff.view.draw_pianoroll_with_velocities(midi_piece=piece, figres=figres)
    return fig


if __name__ == "__main__":
    spec = plot_spectrogram()
    pianoroll = plot_pianoroll()
    plt.show()
    # spec.savefig("data/img/spectrogram.png")
    # pianoroll.savefig("data/img/pianoroll.png")
