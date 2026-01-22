import ollama

BASE_PROMPT = """
Tu es un analyste conversationnel spécialisé dans la relation client assurance (MAIF).
Tu produis des résumés factuels, neutres et exploitables pour l'amélioration des processus internes.

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
- N'invente rien. Si une info manque: Non précisé dans l'appel.
- Interlocuteur = personne qui parle à l'oral.
- Assuré = personne concernée par le contrat / sinistre.
- Ne produis AUCUN autre titre, AUCUN commentaire, AUCUN paragraphe hors format.
- Il est important de noter que l'interlocuteur peut appeler au nom du sociétaire donc il est ESSENTIEL de faire la distinction, le cas échéant

Transcription :
--------------------------------
{transcript}
--------------------------------
"""


def summarize(*, transcript: str) -> str:
    print("Starting summary call")
    prompt = BASE_PROMPT.format(transcript=transcript)
    response = ollama.generate(model="llama3", prompt=prompt)["response"]
    print("Summary call completed")
    return response
