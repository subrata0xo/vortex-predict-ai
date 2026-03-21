"""
Unified model builder — picks LSTM or Hybrid based on model_type.
"""

from models.lstm import build_lstm
from models.hybrid import build_hybrid


def build_model(model_type: str = "hybrid", **kwargs):
    """
    Build a cyclone prediction model.

    Args:
        model_type: "lstm" or "hybrid"
        **kwargs: model hyperparameters passed to the builder

    Returns:
        nn.Module ready for training or inference
    """
    builders = {
        "lstm":   build_lstm,
        "hybrid": build_hybrid,
    }
    if model_type not in builders:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {list(builders.keys())}")
    return builders[model_type](**kwargs)


__all__ = ["build_model", "build_lstm", "build_hybrid"]
