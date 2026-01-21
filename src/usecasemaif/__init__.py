from usecasemaif.config import Settings, get_settings
from usecasemaif.models import (
    Segment,
    Speaker,
    TranscriptionRequest,
    TranscriptionResult,
)
from usecasemaif.pipeline import TranscriptionPipeline

__all__ = [
    "Settings",
    "get_settings",
    "Segment",
    "Speaker",
    "TranscriptionRequest",
    "TranscriptionResult",
    "TranscriptionPipeline",
]


def main() -> None:
    """Point d'entrée CLI du package."""
    print("UseCaseMaif - Pipeline de transcription audio")
