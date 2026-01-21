from pathlib import Path
from pydantic import BaseModel, Field, field_validator


class Speaker(BaseModel):
    """Représente un locuteur identifié dans l'audio."""

    id: str = Field(..., description="Identifiant unique du locuteur (ex: SPEAKER_00)")
    label: str | None = Field(
        default=None, description="Label personnalisé du locuteur (ex: Agent, Client)"
    )

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Speaker):
            return False
        return self.id == other.id


class Segment(BaseModel):
    """Représente un segment de transcription avec timing et locuteur."""

    start: float = Field(..., ge=0, description="Temps de début en secondes")
    end: float = Field(..., ge=0, description="Temps de fin en secondes")
    text: str = Field(..., description="Texte transcrit du segment")
    speaker: Speaker | None = Field(
        default=None, description="Locuteur du segment (si diarization activée)"
    )
    confidence: float | None = Field(
        default=None, ge=0, le=1, description="Score de confiance de la transcription"
    )

    @field_validator("end")
    @classmethod
    def end_must_be_after_start(cls, v: float, info) -> float:
        start = info.data.get("start")
        if start is not None and v < start:
            raise ValueError("end doit être supérieur ou égal à start")
        return v

    @property
    def duration(self) -> float:
        """Durée du segment en secondes."""
        return self.end - self.start


class TranscriptionResult(BaseModel):
    """Résultat complet d'une transcription audio."""

    audio_path: Path = Field(..., description="Chemin du fichier audio source")
    language: str = Field(..., description="Code langue détectée (ex: fr, en)")
    duration: float = Field(..., ge=0, description="Durée totale de l'audio en secondes")
    segments: list[Segment] = Field(
        default_factory=list, description="Liste des segments transcrits"
    )
    speakers: list[Speaker] = Field(
        default_factory=list, description="Liste des locuteurs identifiés"
    )

    @property
    def full_text(self) -> str:
        """Retourne la transcription complète sans timestamps."""
        return " ".join(segment.text.strip() for segment in self.segments)

    @property
    def text_by_speaker(self) -> dict[str, list[str]]:
        """Retourne les textes groupés par locuteur."""
        result: dict[str, list[str]] = {}
        for segment in self.segments:
            speaker_id = segment.speaker.id if segment.speaker else "UNKNOWN"
            if speaker_id not in result:
                result[speaker_id] = []
            result[speaker_id].append(segment.text.strip())
        return result

    def to_srt(self) -> str:
        """Exporte la transcription au format SRT."""
        lines = []
        for i, segment in enumerate(self.segments, start=1):
            start_time = self._format_srt_time(segment.start)
            end_time = self._format_srt_time(segment.end)
            speaker_prefix = f"[{segment.speaker.id}] " if segment.speaker else ""
            lines.append(f"{i}")
            lines.append(f"{start_time} --> {end_time}")
            lines.append(f"{speaker_prefix}{segment.text.strip()}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Formate un temps en secondes au format SRT (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class TranscriptionRequest(BaseModel):
    """Requête de transcription audio."""

    audio_path: Path = Field(..., description="Chemin du fichier audio à transcrire")
    language: str | None = Field(
        default=None, description="Code langue (auto-détection si None)"
    )
    enable_diarization: bool = Field(
        default=True, description="Activer la diarization (identification des locuteurs)"
    )
    min_speakers: int | None = Field(
        default=None, ge=1, description="Nombre minimum de locuteurs attendus"
    )
    max_speakers: int | None = Field(
        default=None, ge=1, description="Nombre maximum de locuteurs attendus"
    )
    model_size: str = Field(
        default="large-v3",
        description="Taille du modèle Whisper (tiny, base, small, medium, large-v3)",
    )
    device: str = Field(
        default="cuda", description="Device pour l'inférence (cuda, cpu)"
    )
    compute_type: str = Field(
        default="float16",
        description="Type de calcul (float16, float32, int8)",
    )

    @field_validator("audio_path")
    @classmethod
    def validate_audio_path(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Le fichier audio n'existe pas: {v}")
        if not v.is_file():
            raise ValueError(f"Le chemin n'est pas un fichier: {v}")
        return v

    @field_validator("max_speakers")
    @classmethod
    def validate_max_speakers(cls, v: int | None, info) -> int | None:
        min_speakers = info.data.get("min_speakers")
        if v is not None and min_speakers is not None and v < min_speakers:
            raise ValueError("max_speakers doit être >= min_speakers")
        return v
