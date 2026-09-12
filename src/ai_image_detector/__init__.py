"""AI Image Detector - a small CNN that classifies images as AI-generated or human-made."""

from .model import AIImageDetector, CLASS_NAMES, IMAGE_SIZE, build_transform

__all__ = ["AIImageDetector", "CLASS_NAMES", "IMAGE_SIZE", "build_transform"]
__version__ = "1.0.0"
