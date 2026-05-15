from __future__ import annotations


class QuantumCafeError(Exception):
    """Base exception for the project."""


class DatasetNotFoundError(QuantumCafeError):
    """Raised when a dataset directory or file is missing."""


class UnknownLabelError(QuantumCafeError):
    """Raised when a folder/label name cannot be mapped to a unified class."""


class ModelNotFoundError(QuantumCafeError):
    """Raised when a model name is not in the registry."""


class CheckpointNotFoundError(QuantumCafeError):
    """Raised when a checkpoint file does not exist."""
