"""Local Russian article-to-MP3 conversion."""

import warnings

__version__ = "0.1.0"

# Silence two noisy third-party warnings emitted while loading Silero on CPU.
# Both originate outside this project and cannot be fixed at their source, so we
# filter them narrowly (by category and message) rather than blanket-ignoring:
#   * PyTorch warns that NumPy is unavailable, but synthesis never uses the NumPy
#     bridge here (PCM is produced via ``tensor.tolist()``), so it is pure noise.
#   * The Silero model package, compiled through ``torch.package`` at load time,
#     contains a non-raw regex literal that Python 3.12 flags as a SyntaxWarning.
# Registered here so they are active before any submodule imports ``torch``.
warnings.filterwarnings(
    "ignore",
    message="Failed to initialize NumPy",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="invalid escape sequence",
    category=SyntaxWarning,
)
