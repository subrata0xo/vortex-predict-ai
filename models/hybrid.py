"""
Hybrid model factory — LSTM + ConvLSTM + Transformer.
"""

from src.model import HybridCycloneModel


def build_hybrid(track_input: int = 17, track_hidden: int = 128,
                 track_layers: int = 2, era5_channels: int = 8,
                 era5_hidden: int = 32, fusion_dim: int = 192,
                 n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.2) -> HybridCycloneModel:
    """Build a HybridCycloneModel with configurable hyperparameters."""
    return HybridCycloneModel(
        track_input=track_input,
        track_hidden=track_hidden,
        track_layers=track_layers,
        era5_channels=era5_channels,
        era5_hidden=era5_hidden,
        fusion_dim=fusion_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    )


__all__ = ["HybridCycloneModel", "build_hybrid"]
