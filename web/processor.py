import time
import wave
import os
import tempfile
import uuid
import subprocess
import shutil
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()


def get_wav_metadata(filepath):
    """
    Extract real metadata from a WAV file.
    """
    try:
        with wave.open(filepath, 'rb') as wav_file:
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
                "filename": os.path.basename(filepath),
                "duration": duration_str,
                "sample_rate": sample_rate_str
            }
    except Exception as e:
        return {
            "filename": os.path.basename(filepath),
            "duration": "N/A",
            "sample_rate": "N/A"
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
    with open(temp_filepath, 'wb') as f:
        f.write(audio_data)
    
    return temp_filepath


def transcribe_with_whisperx(audio_filepath):
    # Create a temp directory for whisperx output
    output_dir = tempfile.mkdtemp()
    
    try:
        # Run whisperx via CLI
        cmd = [
            "whisperx",
            audio_filepath,
            "--compute_type", "int8",
            "--diarize",
            "--min_speakers", "2",
            "--max_speakers", "2",
            "--model", "large-v3",
            "--language", "fr",
            "--output_dir", output_dir,
            "--hf_token", os.getenv("HF_TOKEN")
        ]
        print(f"Running whisperx: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Whisperx error: {result.stderr}")
            return None
        
        # Read the output txt file
        audio_basename = os.path.splitext(os.path.basename(audio_filepath))[0]
        txt_output_path = os.path.join(output_dir, f"{audio_basename}.txt")
        
        if os.path.exists(txt_output_path):
            with open(txt_output_path, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
            print(f"Transcript: {transcript}")
            return transcript
        else:
            print(f"Output file not found: {txt_output_path}")
            return None
            
    finally:
        # Clean up temp output directory
        shutil.rmtree(output_dir, ignore_errors=True)


def process_wav(filepath, first_speaker='maif'):
    # Get real metadata from the WAV file
    metadata = get_wav_metadata(filepath)
    
    # Read the audio file
    with open(filepath, 'rb') as f:
        audio_data = f.read()
    
    # Save to temp file with UUID
    temp_audio_path = save_audio_to_temp(audio_data)
    
    try:
        # Transcribe with whisperx
        transcript = transcribe_with_whisperx(temp_audio_path)
        
        if transcript is None:
            transcript = "Erreur lors de la transcription"
        
        print(f"Transcription finale: {transcript}")
        
        # Return placeholder response to frontend
        return {
            "transcript": transcript,
            "emotions": {
                "primary": "En cours d'analyse...",
                "confidence": 0
            },
            "metadata": metadata
        }
    finally:
        # Clean up temp audio file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
