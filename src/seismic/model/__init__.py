"""Modèles physiques du sous-sol."""

from seismic.model.velocity_model import (
    VelocityModel,
    build_synthetic_velocity_model,
)

__all__ = [
    "VelocityModel",
    "build_synthetic_velocity_model",
]
