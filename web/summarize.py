from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

SYSTEM_PROMPT = """
Tu es un analyste conversationnel spécialisé dans la relation client assurance (MAIF).
Tu produis des résumés factuels, neutres et exploitables pour l'amélioration des processus internes.
"""

USER_PROMPT_TEMPLATE = """
Analyse la transcription ci-dessous.

Tu dois produire UNIQUEMENT ce format, sans aucun autre texte, sans markdown.

Format EXACT à respecter :

PROBLEMATIQUE_PRINCIPALE:
(une phrase)

RESUME_GLOBAL:
(2 à 4 phrases)

RESUME_POUR_ASSUREUR:
- Interlocuteur :
- Assuré :
- Contrat / bien / situation concerné(e) :
- Événement (si sinistre) :
- Décisions prises :
- En attente / à valider :
- Informations expliquées (non décidées) :
- Options / garanties refusées :
- Garanties / couvertures évoquées :
- Montants / franchises :
- Pièces / preuves demandées :
- Actions à réaliser :

RESUME_POUR_ASSURE:
(texte simple sur ce qui est choisi, ce qui reste à faire, prochaines étapes)

Règles:
- Utilise uniquement le texte dans la transcription
- N’invente rien. Si une info manque: Non précisé dans l’appel.
- Interlocuteur = personne qui parle à l’oral.
- Assuré = personne concernée par le contrat / sinistre.
- Ne produis AUCUN autre titre, AUCUN commentaire, AUCUN paragraphe hors format.
- Il est important de noter que l'interlocuteur peut appeler au nom du sociétaire donc il est ESSENTIEL de faire la distinction, le cas échéant

Transcription :
--------------------------------
{transcript}
--------------------------------
"""

def summarize_call(model, transcript: str, max_new_tokens: int = 1024) -> str:
  messages = [
    {
      "role": "system",
      "content": SYSTEM_PROMPT.strip()
    },
    {
      "role": "user",
      "content": USER_PROMPT_TEMPLATE.format(transcript=transcript).strip()
    }
  ]

  prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
  )

  inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

  with torch.no_grad():
    output_ids = model.generate(
      **inputs,
      max_new_tokens=max_new_tokens,
      temperature=0.3,
      top_p=0.9,
      do_sample=True
    )

    generated = output_ids[0][inputs.input_ids.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
 

def main(transcript: str, model_name: str = "Qwen/Qwen3-4B-Instruct-2507") -> str:
  global tokenizer, model

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    attn_implementation="flash_attention_2"
  ).to(device)

  model.eval()

  print("Synthétisation de l'appel...")
  time_start = time.time()
  summary = summarize_call(model, transcript)
  time_end = time.time()
  print(f"Synthétisation completée en {time_end - time_start:.2f} secondes.")
  return summary