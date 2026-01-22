import os

import audiofile
import numpy as np
from rVADfast import rVADfast


def normalize_audio(signal: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val
    return signal.astype(np.float32)


def preprocess_audio(
    *, file_path: str, mono: bool = True, target_sr: int = 16_000
) -> bool:
    """
    Apply rVADfast to an audio file and overwrite it with speech-only audio.
    Falls back to original audio if no speech is detected.

    Returns
    -------
    bool
        True if VAD speech was used, False if fallback occurred.
    """

    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)

    # ======== LOAD AUDIO ========
    waveform, sampling_rate = audiofile.read(file_path)

    # Force mono
    if mono and waveform.ndim > 1:
        waveform = waveform.mean(axis=0)

    waveform = waveform.astype(np.float32)

    # Resample if needed
    if target_sr is not None and sampling_rate != target_sr:
        import librosa

        waveform = librosa.resample(
            waveform, orig_sr=sampling_rate, target_sr=target_sr
        )
        sampling_rate = target_sr

    # Normalize BEFORE VAD
    waveform = normalize_audio(waveform)

    # ======== INIT VAD ========
    vad = rVADfast()

    try:
        vad_labels, vad_timestamps = vad(waveform, sampling_rate)
    except Exception as e:
        print(f"[VAD WARNING] rVAD failed ({e}), keeping original audio.")
        audiofile.write(file_path, waveform, sampling_rate)
        return False

    # ======== COLLECT SPEECH SEGMENTS ========
    speech_segments = []

    for label, ts in zip(vad_labels, vad_timestamps):
        if label != "speech":
            continue

        if not isinstance(ts, (list, tuple)) or len(ts) != 2:
            continue

        start, end = ts
        if np.isnan(start) or np.isnan(end) or end <= start:
            continue

        start_sample = int(start * sampling_rate)
        end_sample = int(end * sampling_rate)

        if end_sample > start_sample:
            speech_segments.append(waveform[start_sample:end_sample])

    # ======== FALLBACK IF NO SPEECH ========
    if not speech_segments:
        print(f"[VAD INFO] No speech detected in {file_path}, keeping original audio.")
        audiofile.write(file_path, waveform, sampling_rate)
        return False

    # ======== CONCAT & NORMALIZE ========
    speech_audio = np.concatenate(speech_segments)
    speech_audio = normalize_audio(speech_audio)

    # ======== OVERWRITE FILE ========
    audiofile.write(file_path, speech_audio, sampling_rate)
    return True
