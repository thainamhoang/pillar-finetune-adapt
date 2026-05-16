"""Pillar: Medical imaging deep learning framework.

Keep package import lightweight.

Subpackages are intentionally not eagerly imported here because some workflows
only need datasets/models, while others may pull in optional metric/loss
dependencies that are not required for smoke tests or feature extraction.
"""

__version__ = "0.2.0"

__all__ = ["datasets", "models", "engines", "losses", "metrics", "augmentations", "utils"]
