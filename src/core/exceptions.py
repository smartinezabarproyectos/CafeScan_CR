from __future__ import annotations

class QuantumCafeError(Exception):
    pass

class DatasetNotFoundError(QuantumCafeError):
    pass

class UnknownLabelError(QuantumCafeError):
    pass

class ModelNotFoundError(QuantumCafeError):
    pass

class CheckpointNotFoundError(QuantumCafeError):
    pass
