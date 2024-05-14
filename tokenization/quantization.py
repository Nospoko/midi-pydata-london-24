import yaml
import numpy as np
import pandas as pd
from fortepyan import MidiPiece


class MidiQuantizer:
    """
    A class for quantizing MIDI data into discrete bins.

    Attributes:
    - n_dstart_bins: Number of bins for delta start time.
    - n_duration_bins: Number of bins for duration.
    - n_velocity_bins: Number of bins for velocity.
    - keys: The keys used for quantization.
    """

    def __init__(
        self,
        n_dstart_bins: int = 3,
        n_duration_bins: int = 3,
        n_velocity_bins: int = 3,
    ):
        """
        Initialize the MidiQuantizer with specified bins.

        Parameters:
        - n_dstart_bins (int): Number of bins for delta start time.
        - n_duration_bins (int): Number of bins for duration.
        - n_velocity_bins (int): Number of bins for velocity.
        """
        self.keys = ["pitch", "dstart_bin", "duration_bin", "velocity_bin"]
        self.n_dstart_bins = n_dstart_bins
        self.n_duration_bins = n_duration_bins
        self.n_velocity_bins = n_velocity_bins
        self._build()

    def __rich_repr__(self):
        """
        Provide a representation of the MidiQuantizer.
        """
        yield "RelativeTimeQuantizer"
        yield "n_dstart_bins", self.n_dstart_bins
        yield "n_duration_bins", self.n_duration_bins
        yield "n_velocity_bins", self.n_velocity_bins

    def quantize_piece(self, piece: MidiPiece) -> MidiPiece:
        """
        Quantize a MIDI piece.

        Parameters:
        - piece (MidiPiece): The MIDI piece to quantize.

        Returns:
        - MidiPiece: The quantized MIDI piece.
        """
        # Copy the DataFrame to avoid overwriting
        df = piece.df.copy()
        source = dict(piece.source) | {"quantized": True}

        # Perform the quantization
        df = self.quantize_frame(df)
        df = self.apply_quantization(df)
        out = MidiPiece(df=df, source=source)
        return out

    def inject_quantization_features(self, piece: MidiPiece) -> MidiPiece:
        """
        Inject quantization features into a MIDI piece.

        Parameters:
        - piece (MidiPiece): The MIDI piece to inject features into.

        Returns:
        - MidiPiece: The MIDI piece with injected quantization features.
        """
        # Copy the DataFrame to avoid overwriting
        df = piece.df.copy()
        source = dict(piece.source) | {"quantized": True}

        # Perform the quantization
        df = self.quantize_frame(df)
        out = MidiPiece(df=df, source=source)
        return out

    def _build(self):
        """
        Build the quantizer by loading bin edges and building decoders.
        """
        self._load_bin_edges()
        self._build_dstart_decoder()
        self._build_duration_decoder()
        self._build_velocity_decoder()

    def _load_bin_edges(self):
        """
        Load bin edges from a YAML file.
        """
        artifacts_path = "midi_quantization_artifacts/bin_edges.yaml"
        with open(artifacts_path, "r") as f:
            bin_edges = yaml.safe_load(f)

        self.dstart_bin_edges = bin_edges["dstart"][self.n_dstart_bins]
        self.duration_bin_edges = bin_edges["duration"][self.n_duration_bins]
        self.velocity_bin_edges = bin_edges["velocity"][self.n_velocity_bins]

    def quantize_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Quantize the DataFrame columns into bins.

        Parameters:
        - df (pd.DataFrame): The DataFrame to quantize.

        Returns:
        - pd.DataFrame: The quantized DataFrame.
        """
        next_start = df.start.shift(-1)
        dstart = next_start - df.start
        df["dstart_bin"] = np.digitize(dstart.fillna(0), self.dstart_bin_edges) - 1
        df["duration_bin"] = np.digitize(df.duration, self.duration_bin_edges) - 1
        df["velocity_bin"] = np.digitize(df.velocity, self.velocity_bin_edges) - 1

        return df

    def quantize_velocity(self, velocity: np.array) -> np.array:
        """
        Quantize the velocity values into bins.

        Parameters:
        - velocity (np.array): The array of velocity values to quantize.

        Returns:
        - np.array: The quantized velocity values.
        """
        velocity_bins = np.digitize(velocity, self.velocity_bin_edges) - 1
        quantized_velocity = np.array([self.bin_to_velocity[v_bin] for v_bin in velocity_bins])
        return quantized_velocity

    def apply_quantization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the quantization to the DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame to apply quantization to.

        Returns:
        - pd.DataFrame: The DataFrame with quantized values.
        """
        quant_dstart = df.dstart_bin.map(lambda it: self.bin_to_dstart[it])
        quant_duration = df.duration_bin.map(lambda it: self.bin_to_duration[it])
        df["start"] = quant_dstart.cumsum().shift(1).fillna(0)
        df["end"] = df.start + quant_duration
        df["duration"] = quant_duration
        df["velocity"] = df.velocity_bin.map(lambda it: self.bin_to_velocity[it])
        return df

    def _build_duration_decoder(self):
        """
        Build the decoder for duration bins.
        """
        self.bin_to_duration = []
        for it in range(1, len(self.duration_bin_edges)):
            duration = (self.duration_bin_edges[it - 1] + self.duration_bin_edges[it]) / 2
            self.bin_to_duration.append(duration)

        last_duration = 2 * self.duration_bin_edges[-1]
        self.bin_to_duration.append(last_duration)

    def _build_dstart_decoder(self):
        """
        Build the decoder for delta start time bins.
        """
        self.bin_to_dstart = []
        for it in range(1, len(self.dstart_bin_edges)):
            dstart = (self.dstart_bin_edges[it - 1] + self.dstart_bin_edges[it]) / 2
            self.bin_to_dstart.append(dstart)

        last_dstart = 2 * self.dstart_bin_edges[-1]
        self.bin_to_dstart.append(last_dstart)

    def _build_velocity_decoder(self):
        """
        Build the decoder for velocity bins.
        """
        # For velocity, the first bin is not going to be evenly populated,
        # skewing towards higher values (who plays with velocity 0?)
        self.bin_to_velocity = [int(0.8 * self.velocity_bin_edges[1])]

        for it in range(2, len(self.velocity_bin_edges)):
            velocity = (self.velocity_bin_edges[it - 1] + self.velocity_bin_edges[it]) / 2
            self.bin_to_velocity.append(int(velocity))

    def make_vocab(self) -> list[str]:
        """
        Create a vocabulary of quantized tokens.

        Returns:
        - list[str]: The vocabulary of quantized tokens.
        """
        vocab = []
        for it, pitch in enumerate(range(21, 109)):
            for jt in range(self.n_duration_bins):
                for kt in range(self.n_dstart_bins):
                    for vt in range(self.n_velocity_bins):
                        key = f"{kt}_{jt}_{vt}_{pitch}"
                        vocab.append(key)

        return vocab
