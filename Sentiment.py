import subprocess
import whisper
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

model = whisper.load_model("tiny", device=device)


result = model.transcribe(
    "processed_segments/Ciel mon mari.wav",
    language="fr",
    fp16=True
)

#print(result2["text"])

transcription = result

prompt = f"""
Tu es un analyste de satisfaction client.
Réponds UNIQUEMENT en français et UNIQUEMENT en JSON.

repond uniquement dans ce format :
{{
  "sentiment": "satisfait | neutre | insatisfait",
  "note": nombre entre 0 et 10,
  "justification": "texte court en français"
}}
Texte : {transcription}
"""

result = subprocess.run(
    ["ollama", "run", "llama3"],
    input=prompt,
    text=True,
    capture_output=True
)

print(result.stdout)
