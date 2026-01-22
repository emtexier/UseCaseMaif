import argparse
import io
import os
import shutil
import tempfile
import uuid
import wave

import torch
from dotenv import load_dotenv
from whisperx.transcribe import transcribe_task

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()


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


def rename_speakers(*, text: str, first_speaker: str) -> str:
    SPEAKERS_MAPPING = {
        "SPEAKER_00": "Opérateur MAIF" if first_speaker == "maif" else "Sociétaire",
        "SPEAKER_01": "Sociétaire" if first_speaker == "maif" else "Opérateur MAIF",
    }

    for original_speaker, speaker_replacement in SPEAKERS_MAPPING.items():
        text = text.replace(original_speaker, speaker_replacement)

    return text


def call_transcribe_task(
    audio_filepath: str,
    output_dir: str,
):
    args = {
        "audio": [audio_filepath],
        # model / runtime
        "model": "large-v2",
        "model_cache_only": False,
        "model_dir": None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "device_index": 0,
        "batch_size": 8,
        "compute_type": "int8",
        # output / logging
        "output_dir": output_dir,
        "output_format": "all",
        "verbose": True,
        "log_level": None,
        # task / language
        "task": "transcribe",
        "language": "fr",
        # alignment
        "align_model": None,
        "interpolate_method": "nearest",
        "no_align": False,
        "return_char_alignments": False,
        # VAD
        "vad_method": "pyannote",
        "vad_onset": 0.5,
        "vad_offset": 0.363,
        "chunk_size": 30,
        # diarization
        "diarize": True,
        "min_speakers": 2,
        "max_speakers": 2,
        "diarize_model": "pyannote/speaker-diarization-3.1",
        "speaker_embeddings": False,
        # decoding
        "temperature": 0.0,
        "best_of": 5,
        "beam_size": 5,
        "patience": 1.0,
        "length_penalty": 1.0,
        "suppress_tokens": "-1",
        "suppress_numerals": False,
        # prompts / fp16
        "initial_prompt": None,
        "condition_on_previous_text": False,
        "fp16": True,
        # fallback
        "temperature_increment_on_fallback": 0.2,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        # formatting
        "max_line_width": None,
        "max_line_count": None,
        "highlight_words": False,
        "segment_resolution": "sentence",
        # performance
        "threads": 0,
        # auth
        "hf_token": os.getenv("HF_TOKEN"),
        "print_progress": False,
    }

    parser = argparse.ArgumentParser()

    transcribe_task(args, parser)


def transcribe_with_whisperx(audio_filepath: str, first_speaker="maif"):
    # Create a temp directory for whisperx output
    output_dir = tempfile.mkdtemp()

    try:
        try:
            # Run whisperx via CLI
            call_transcribe_task(audio_filepath=audio_filepath, output_dir=output_dir)
        except Exception as e:
            print("Error WhisperX:", e)
            return None

        # Read the output txt file
        audio_basename = os.path.splitext(os.path.basename(audio_filepath))[0]
        txt_output_path = os.path.join(output_dir, f"{audio_basename}.txt")

        if os.path.exists(txt_output_path):
            with open(txt_output_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
                transcript = rename_speakers(
                    text=transcript, first_speaker=first_speaker
                )
            with open(txt_output_path, "w", encoding="utf-8") as f:
                f.write(transcript)
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
