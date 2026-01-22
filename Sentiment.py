import subprocess
import whisper
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("tiny", device=device)

result = model.transcribe(
    "processed_segments/Ciel mon mari.wav",
    language="fr",
    fp16=(device == "cuda")
)

transcription = result["text"]

prompt = f"""
Tu es un analyste de satisfaction client.

Réponds STRICTEMENT en JSON valide.
AUCUN texte avant ou après.

Format exact :
{{"sentiment":"satisfait|neutre|insatisfait","note":0-10,"justification":"texte court"}}

Texte client :
\"\"\"{transcription}\"\"\"
"""

result = subprocess.run(
    ["ollama", "run", "llama3"],
    input=prompt,
    text=True,
    capture_output=True,
    encoding="utf-8"
)

print(result.stdout)
