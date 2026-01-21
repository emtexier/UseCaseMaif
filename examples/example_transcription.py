"""
Exemple d'utilisation du pipeline de transcription audio.

Ce script montre comment utiliser le TranscriptionPipeline pour transcrire
un fichier audio avec identification des locuteurs (diarization).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from usecasemaif.pipeline import TranscriptionPipeline

# Charger les variables d'environnement depuis .env
load_dotenv()


def main():
    # Récupérer le token HuggingFace depuis les variables d'environnement
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError(
            "Le token HUGGINGFACE_TOKEN n'est pas défini dans le fichier .env"
        )

    # Chemin vers le dossier des fichiers audio
    project_root = Path(__file__).parent.parent
    audio_dir = project_root / "processed_segments"

    # Choisir un fichier audio à transcrire
    audio_file = audio_dir / "Je brise la glace.wav"

    if not audio_file.exists():
        print(f"Fichier audio non trouvé: {audio_file}")
        print("Fichiers disponibles:")
        for f in audio_dir.iterdir():
            print(f"  - {f.name}")
        return

    print(f"Transcription de: {audio_file.name}")
    print("-" * 50)

    # Initialiser le pipeline
    pipeline = TranscriptionPipeline(
        model_size="medium",  # Options: tiny, base, small, medium, large-v3
        hf_token=hf_token,
    )

    # Transcrire le fichier audio
    result = pipeline.transcribe_file(
        audio_path=audio_file,
        language=None,  # Auto-détection de la langue
        enable_diarization=True,  # Activer l'identification des locuteurs
        min_speakers=None,  # Laisser le modèle deviner
        max_speakers=None,
    )

    # Afficher les résultats
    print(f"\nLangue détectée: {result.language}")
    print(f"Durée: {result.duration:.2f} secondes")
    print(f"Nombre de segments: {len(result.segments)}")
    print(f"Nombre de locuteurs: {len(result.speakers)}")

    print("\n" + "=" * 50)
    print("TRANSCRIPTION COMPLÈTE")
    print("=" * 50)

    for segment in result.segments:
        speaker_label = f"[{segment.speaker.id}]" if segment.speaker else "[?]"
        timestamp = f"[{segment.start:.2f}s - {segment.end:.2f}s]"
        print(f"{timestamp} {speaker_label}: {segment.text.strip()}")

    # Afficher le texte complet
    print("\n" + "=" * 50)
    print("TEXTE COMPLET")
    print("=" * 50)
    print(result.full_text)

    # Exporter en SRT si besoin
    output_dir = project_root / "processed_segments"
    output_dir.mkdir(exist_ok=True)

    srt_output = output_dir / f"{audio_file.stem}.srt"
    srt_output.write_text(result.to_srt(), encoding="utf-8")
    print(f"\nFichier SRT exporté: {srt_output}")

    # Libérer la mémoire GPU
    pipeline.unload_models()


if __name__ == "__main__":
    main()
