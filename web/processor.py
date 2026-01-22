import io
import os
import shutil
import subprocess
import tempfile
import uuid
import wave

import lightning_fabric.utilities.cloud_io
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()


def patch_lightningfabric_load(path_or_url, map_location, weights_only):
    from pathlib import Path

    import torch

    if not isinstance(path_or_url, (str, Path)):
        # any sort of BytesIO or similar
        return torch.load(
            path_or_url,
            map_location=map_location,  # type: ignore[arg-type] # upstream annotation is not correct
            weights_only=weights_only,
        )
    if str(path_or_url).startswith("http"):
        if weights_only is None:
            weights_only = False

        return torch.hub.load_state_dict_from_url(
            str(path_or_url),
            map_location=map_location,  # type: ignore[arg-type]
            weights_only=weights_only,
        )
    fs = lightning_fabric.utilities.cloud_io.get_filesystem(path_or_url)
    with fs.open(path_or_url, "rb") as f:
        return torch.load(
            f,
            map_location=map_location,  # type: ignore[arg-type]
            # monkeypatching is here
            weights_only=False,  # weights_only,
        )


lightning_fabric.utilities.cloud_io._load = patch_lightningfabric_load


def get_wav_metadata(*, audio_data, filename):
    """
    Extract real metadata from WAV audio data.
    Args:
        audio_data: bytes object containing WAV audio data
        filename: original filename for reference
    """
    try:
        # Create a file-like object from bytes
        audio_stream = io.BytesIO(audio_data)

        with wave.open(audio_stream, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            duration = n_frames / sample_rate

            # Format duration
            if duration >= 60:
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                duration_str = f"{minutes}m {seconds}s"
            else:
                duration_str = f"{duration:.1f}s"

            # Format sample rate
            if sample_rate >= 1000:
                sample_rate_str = f"{sample_rate / 1000:.1f}kHz"
            else:
                sample_rate_str = f"{sample_rate}Hz"

            return {
                "filename": filename,
                "duration": duration_str,
                "sample_rate": sample_rate_str,
            }
    except Exception:
        return {
            "filename": filename,
            "duration": "N/A",
            "sample_rate": "N/A",
        }


def save_audio_to_temp(audio_data):
    """
    Save audio data to a temporary file with a UUID filename for security.
    Returns the path to the temporary file.
    """
    # Create a unique filename with UUID
    unique_filename = f"{uuid.uuid4()}.wav"

    # Create temp directory if it doesn't exist
    temp_dir = tempfile.gettempdir()
    temp_filepath = os.path.join(temp_dir, unique_filename)

    # Write audio data to temp file
    with open(temp_filepath, "wb") as f:
        f.write(audio_data)

    return temp_filepath


def rename_speakers(*, filepath: str, first_speaker: str) -> None:
    SPEAKERS_MAPPING = {
        "SPEAKER_00": "Opérateur MAIF" if first_speaker == "maif" else "Sociétaire",
        "SPEAKER_01": "Sociétaire" if first_speaker == "maif" else "Opérateur MAIF",
    }

    text = ""
    with open(file=filepath, mode="r") as f:
        text = f.read()

    for original_speaker, speaker_replacement in SPEAKERS_MAPPING.items():
        text = text.replace(original_speaker, speaker_replacement)

    with open(file=filepath, mode="w") as f:
        f.write(text)


def transcribe_with_whisperx(audio_filepath, first_speaker="maif"):
    # Create a temp directory for whisperx output
    output_dir = tempfile.mkdtemp()

    try:
        # Run whisperx via CLI
        cmd = [
            "whisperx",
            audio_filepath,
            "--compute_type",
            "int8",
            "--diarize",
            "--min_speakers",
            "2",
            "--max_speakers",
            "2",
            "--model",
            "large-v2",
            "--language",
            "fr",
            "--output_dir",
            output_dir,
            "--hf_token",
            os.getenv("HF_TOKEN"),
        ]
        print(f"Running whisperx: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Whisperx error: {result.stderr}")
            return None

        # Read the output txt file
        audio_basename = os.path.splitext(os.path.basename(audio_filepath))[0]
        txt_output_path = os.path.join(output_dir, f"{audio_basename}.txt")

        if os.path.exists(txt_output_path):
            with open(txt_output_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
                rename_speakers(filepath=txt_output_path, first_speaker=first_speaker)
            print(f"Transcript: {transcript}")
            return transcript
        else:
            print(f"Output file not found: {txt_output_path}")
            return None

    finally:
        # Clean up temp output directory
        shutil.rmtree(output_dir, ignore_errors=True)


def process_wav(audio_data, first_speaker="maif"):
    # Save to temp file with UUID
    temp_audio_path = save_audio_to_temp(audio_data)

    # Get real metadata from the WAV file
    metadata = get_wav_metadata(audio_data=audio_data, filename=temp_audio_path)

    try:
        # Transcribe with whisperx
        transcript = transcribe_with_whisperx(
            temp_audio_path, first_speaker=first_speaker
        )

        if transcript is None:
            transcript = "Erreur lors de la transcription"

        print(f"Transcription finale: {transcript}")

        # Return placeholder response to frontend
        return {
            "transcript": transcript,
            "emotions": {"primary": "En cours d'analyse...", "confidence": 0},
            "metadata": metadata,
        }
    finally:
        # Clean up temp audio file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
