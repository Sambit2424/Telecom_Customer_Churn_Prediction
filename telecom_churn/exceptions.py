class ModelLoadError(Exception):
    """Raised when a serialized machine learning model cannot be loaded."""


class PredictionError(Exception):
    """Raised when there is an error during prediction."""


class ValidationError(Exception):
    """Raised when incoming request data fails validation."""
