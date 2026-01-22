import time

def process_wav(filepath):
    """
    Placeholder function to process the WAV file.
    In the future, this will call the Nemo ASR model and other analysis tools.
    """
    # Simulate processing time
    time.sleep(1.5)
    
    # Mock Data - To be replaced by actual analysis
    return {
        "transcript": "Bonjour, je vous appelle pour déclarer un sinistre concernant mon véhicule.",
        "emotions": {
            "primary": "Neutre",
            "confidence": 85
        },
        "metadata": {
            "duration": "12s",
            "sample_rate": "44.1kHz"
        }
    }
