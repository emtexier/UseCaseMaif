import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration globale de l'application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Chemins
    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent,
        description="Répertoire racine du projet",
    )
    audio_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "sons",
        description="Répertoire des fichiers audio",
    )
    output_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "output",
        description="Répertoire de sortie",
    )

    # Configuration Whisper
    whisper_model_size: str = Field(
        default="large-v3",
        description="Taille du modèle Whisper",
    )
    whisper_device: str = Field(
        default="cuda",
        description="Device pour l'inférence",
    )
    whisper_compute_type: str = Field(
        default="float16",
        description="Type de calcul",
    )

    # HuggingFace (pour diarization)
    hf_token: str | None = Field(
        default=None,
        description="Token HuggingFace pour pyannote",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Niveau de log",
    )

    def ensure_directories(self) -> None:
        """Crée les répertoires nécessaires s'ils n'existent pas."""
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Retourne une instance des settings."""
    return Settings()
