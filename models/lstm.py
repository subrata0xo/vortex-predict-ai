"""
LSTM baseline model factory.
"""

from src.model import CycloneLSTM


def build_lstm(input_size: int = 17, hidden_size: int = 128,
               num_layers: int = 2, dropout: float = 0.2) -> CycloneLSTM:
    """Build a CycloneLSTM model with configurable hyperparameters."""
    return CycloneLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )


__all__ = ["CycloneLSTM", "build_lstm"]
