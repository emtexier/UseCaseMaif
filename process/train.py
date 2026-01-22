import whisper
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = whisper.load_model("tiny", device=device)

result = model.transcribe(
    "processed_segments/Ciel mon mari.wav",
    language="fr",
    fp16=True
)