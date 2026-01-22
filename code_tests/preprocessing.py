import os

import audiofile
import numpy as np
from pydub import AudioSegment
from rVADfast import rVADfast

# ======== CONFIG ========
INPUT_DIR = "sons"  # dossier contenant les fichiers mp4/wav
OUTPUT_DIR = "processed_segments"
TARGET_SR = 16000  # fréquence standard pour la parole
MONO = True  # on travaille en mono

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======== FONCTIONS UTILES ========
def convert_to_wav(file_path, target_sr=16000, mono=True):
    """Convertit un fichier audio en WAV mono / target_sr."""
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1 if mono else audio.channels)
    audio = audio.set_frame_rate(target_sr)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    wav_path = os.path.join(OUTPUT_DIR, f"{base_name}.wav")
    audio.export(wav_path, format="wav")
    return wav_path


def normalize_audio(signal):
    """Normalise un signal entre -1 et 1"""
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val
    return signal


# ======== INITIALISATION VAD ========
vad = rVADfast()

# ======== BATCH PROCESS ========
for file_name in os.listdir(INPUT_DIR):
    file_path = os.path.join(INPUT_DIR, file_name)

    # Ignorer les fichiers cachés ou dossiers
    if not os.path.isfile(file_path):
        continue

    # Convertir si nécessaire
    if not file_name.lower().endswith(".wav"):
        print(f"Conversion {file_name} → WAV")
        wav_path = convert_to_wav(file_path, TARGET_SR, MONO)
    else:
        wav_path = file_path

    # Lire l'audio
    waveform, sampling_rate = audiofile.read(wav_path)

    # Appliquer le VAD
    vad_labels, vad_timestamps = vad(waveform, sampling_rate)

    # Extraire et sauvegarder les segments speech
    base_name = os.path.splitext(file_name)[0]
    segment_count = 0

    for i in range(len(vad_labels)):
        label = vad_labels[i]
        ts = vad_timestamps[i]

        # Vérifier que ts est un tuple/list et pas un float/NaN
        if isinstance(ts, (list, tuple)) and len(ts) == 2:
            start, end = ts
            if np.isnan(start) or np.isnan(end):
                continue
        else:
            # Segment invalide, ignorer
            continue

        if label == "speech":
            start_sample = int(start * sampling_rate)
            end_sample = int(end * sampling_rate)
            segment = waveform[start_sample:end_sample]
            segment = normalize_audio(segment)

            segment_path = os.path.join(
                OUTPUT_DIR, f"{base_name}_segment_{segment_count}.wav"
            )
            audiofile.write(segment_path, segment, sampling_rate)
            print(f"Segment sauvegardé: {segment_path}")
            segment_count += 1

print("Traitement terminé pour tous les fichiers.")
