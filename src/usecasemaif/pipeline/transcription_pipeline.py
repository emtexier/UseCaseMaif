import logging
from pathlib import Path
from typing import Any

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

from usecasemaif.models import (
    Segment,
    Speaker,
    TranscriptionRequest,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)


class TranscriptionPipeline:
    """Pipeline de transcription audio avec diarization utilisant WhisperX."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str | None = None,
        compute_type: str = "float16",
        hf_token: str | None = None,
    ) -> None:
        """
        Initialise le pipeline de transcription.

        Args:
            model_size: Taille du modèle Whisper à utiliser.
            device: Device pour l'inférence (cuda/cpu). Auto-détecté si None.
            compute_type: Type de calcul pour l'inférence.
            hf_token: Token HuggingFace pour la diarization (pyannote).
        """
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._compute_type = compute_type if self._device == "cuda" else "float32"
        self._model_size = model_size
        self._hf_token = hf_token

        self._model: Any | None = None
        self._diarize_model: DiarizationPipeline | None = None
        self._align_model: tuple | None = None
        self._align_metadata: dict | None = None
        self._current_align_language: str | None = None

        logger.info(
            f"Pipeline initialisé - device: {self._device}, "
            f"compute_type: {self._compute_type}, model: {self._model_size}"
        )

    def _load_whisper_model(self) -> Any:
        """Charge le modèle Whisper si nécessaire."""
        if self._model is None:
            logger.info(f"Chargement du modèle Whisper {self._model_size}...")
            self._model = whisperx.load_model(
                self._model_size,
                self._device,
                compute_type=self._compute_type,
            )
        return self._model

    def _load_align_model(self, language_code: str) -> tuple:
        """Charge le modèle d'alignement pour une langue donnée."""
        if self._align_model is None or self._current_align_language != language_code:
            logger.info(f"Chargement du modèle d'alignement pour {language_code}...")
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=language_code,
                device=self._device,
            )
            self._current_align_language = language_code
        return self._align_model, self._align_metadata

    def _load_diarization_model(self) -> DiarizationPipeline:
        """Charge le modèle de diarization si nécessaire."""
        if self._diarize_model is None:
            if not self._hf_token:
                raise ValueError(
                    "Un token HuggingFace est requis pour la diarization. "
                    "Acceptez les conditions d'utilisation sur: "
                    "https://huggingface.co/pyannote/speaker-diarization-3.1"
                )
            logger.info("Chargement du modèle de diarization...")
            self._diarize_model = DiarizationPipeline(
                use_auth_token=self._hf_token,
                device=self._device,
            )
        return self._diarize_model

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        """
        Transcrit un fichier audio avec optionnellement la diarization.

        Args:
            request: Requête de transcription contenant les paramètres.

        Returns:
            Résultat de la transcription avec segments et locuteurs.
        """
        audio_path = request.audio_path
        logger.info(f"Début de la transcription de {audio_path}")

        # Charger l'audio
        audio = whisperx.load_audio(str(audio_path))

        # Transcription avec Whisper
        model = self._load_whisper_model()
        transcription_result = model.transcribe(
            audio,
            batch_size=16,
            language=request.language,
        )

        detected_language = transcription_result.get("language", "fr")
        logger.info(f"Langue détectée: {detected_language}")

        # Alignement des mots
        align_model, align_metadata = self._load_align_model(detected_language)
        aligned_result = whisperx.align(
            transcription_result["segments"],
            align_model,
            align_metadata,
            audio,
            self._device,
            return_char_alignments=False,
        )

        # Diarization si activée
        speakers: list[Speaker] = []
        if request.enable_diarization:
            diarize_model = self._load_diarization_model()

            diarize_kwargs = {}
            if request.min_speakers is not None:
                diarize_kwargs["min_speakers"] = request.min_speakers
            if request.max_speakers is not None:
                diarize_kwargs["max_speakers"] = request.max_speakers

            diarize_segments = diarize_model(audio, **diarize_kwargs)
            aligned_result = whisperx.assign_word_speakers(
                diarize_segments, aligned_result
            )

            # Extraire les locuteurs uniques
            speaker_ids = set()
            for segment in aligned_result["segments"]:
                if "speaker" in segment:
                    speaker_ids.add(segment["speaker"])

            speakers = [Speaker(id=sid) for sid in sorted(speaker_ids)]
            logger.info(f"Locuteurs identifiés: {len(speakers)}")

        # Construire les segments
        segments = self._build_segments(aligned_result["segments"], speakers)

        # Calculer la durée totale
        duration = len(audio) / 16000  # WhisperX utilise 16kHz

        result = TranscriptionResult(
            audio_path=audio_path,
            language=detected_language,
            duration=duration,
            segments=segments,
            speakers=speakers,
        )

        logger.info(
            f"Transcription terminée - {len(segments)} segments, "
            f"{len(speakers)} locuteurs, durée: {duration:.2f}s"
        )

        return result

    def _build_segments(
        self, raw_segments: list[dict], speakers: list[Speaker]
    ) -> list[Segment]:
        """Construit les objets Segment à partir des segments bruts."""
        speaker_map = {s.id: s for s in speakers}
        segments: list[Segment] = []

        for raw in raw_segments:
            speaker = None
            if "speaker" in raw and raw["speaker"] in speaker_map:
                speaker = speaker_map[raw["speaker"]]

            segment = Segment(
                start=raw.get("start", 0.0),
                end=raw.get("end", 0.0),
                text=raw.get("text", ""),
                speaker=speaker,
                confidence=raw.get("confidence"),
            )
            segments.append(segment)

        return segments

    def transcribe_file(
        self,
        audio_path: str | Path,
        language: str | None = None,
        enable_diarization: bool = True,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> TranscriptionResult:
        """
        Méthode simplifiée pour transcrire un fichier audio.

        Args:
            audio_path: Chemin vers le fichier audio.
            language: Code langue (auto-détection si None).
            enable_diarization: Activer l'identification des locuteurs.
            min_speakers: Nombre minimum de locuteurs.
            max_speakers: Nombre maximum de locuteurs.

        Returns:
            Résultat de la transcription.
        """
        request = TranscriptionRequest(
            audio_path=Path(audio_path),
            language=language,
            enable_diarization=enable_diarization,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            model_size=self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        return self.transcribe(request)

    def unload_models(self) -> None:
        """Libère les modèles de la mémoire."""
        self._model = None
        self._diarize_model = None
        self._align_model = None
        self._align_metadata = None
        self._current_align_language = None

        if self._device == "cuda":
            torch.cuda.empty_cache()

        logger.info("Modèles déchargés de la mémoire")
