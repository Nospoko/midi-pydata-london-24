import fortepyan as ff
from datasets import load_dataset

from tokenization.quantization import MidiQuantizer


def main():
    quantizer = MidiQuantizer(
        n_dstart_bins=5,
        n_duration_bins=5,
        n_velocity_bins=5,
    )

    dataset = load_dataset("roszcz/maestro-sustain-v2", split="test")
    record = dataset[77]

    piece = ff.MidiPiece.from_huggingface(record=record)
    quantized_piece = quantizer.quantize_piece(piece=piece)

    piece_path = "data/example.mid"
    quantized_path = "data/quantized_example.mid"

    piece.to_midi().write(piece_path)
    quantized_piece.to_midi().write(quantized_path)


if __name__ == "__main__":
    main()
