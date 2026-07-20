class PyTTSError(Exception):
    exit_code = 1


class UsageError(PyTTSError):
    exit_code = 2


class InputError(PyTTSError):
    exit_code = 3


class ConfigError(PyTTSError):
    exit_code = 3


class ModelError(PyTTSError):
    exit_code = 4


class ModelIntegrityError(ModelError):
    """The downloaded or cached model does not match the committed manifest."""


class SynthesisError(PyTTSError):
    exit_code = 5


class AudioError(SynthesisError):
    """PCM conversion, LAME encoding, or atomic output failure."""
