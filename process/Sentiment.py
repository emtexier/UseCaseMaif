import subprocess
import whisper
import torch


def analyse_satisfaction_audio(
    audio_path: str,
    whisper_model_name: str = "tiny",
    llm_model_name: str = "llama3",
    language: str = "fr"
) -> dict:
    """
    Transcrit un fichier audio puis analyse la satisfaction client via un LLM (Ollama).

    :param audio_path: chemin vers le fichier audio (.wav, .mp3, etc.)
    :param whisper_model_name: modèle Whisper à utiliser (tiny, base, small, medium, large)
    :param llm_model_name: modèle Ollama (ex: llama3)
    :param language: langue de transcription
    :return: dictionnaire JSON (sentiment, note, justification)
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Chargement du modèle Whisper
    model = whisper.load_model(whisper_model_name, device=device)

    # Transcription
    result = model.transcribe(
        audio_path,
        language=language,
        fp16=(device == "cuda")
    )

    transcription = result["text"]

    # Prompt pour le LLM
    prompt = f"""
    Tu es un analyste de satisfaction client.

    Réponds STRICTEMENT en JSON valide.
    AUCUN texte avant ou après.

    Format exact :
    {{"sentiment":"satisfait|neutre|insatisfait","note":0-10,"justification":"texte court"}}

    Texte client :
    \"\"\"{transcription}\"\"\"
    """

    # Appel à Ollama
    llm_result = subprocess.run(
        ["ollama", "run", llm_model_name],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8"
    )

    # Retour JSON (texte → dict)
    import json
    return json.loads(llm_result.stdout.strip())
