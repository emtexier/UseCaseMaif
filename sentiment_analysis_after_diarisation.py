from transformers import pipeline
import re
from pathlib import Path

def extract_customer_lines(text, file_format='srt'):
    """
    Extrait uniquement les lignes du client de la transcription SRT
    Format SRT avec [SPEAKER_XX]
    """
    lines = text.split('\n')
    customer_text = []
    
    # Format SRT avec [SPEAKER_XX]
    client_speaker = '[SPEAKER_01]'
    
    # Extraire le texte du client
    for line in lines:
        if client_speaker in line:
            content = line[line.find(']')+1:].strip()
            if content:
                customer_text.append(content)
    
    return ' '.join(customer_text)


def analyze_sentiment_french(text):
    """
    Analyse le sentiment du texte en utilisant un modèle multilingue optimisé pour le français
    """
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=0  # GPU (device=-1 pour CPU)
    )
    
    # Découper le texte en phrases si trop long
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    results = []
    
    for i, sentence in enumerate(sentences, 1):
        if len(sentence) > 10:  # Ignorer les phrases très courtes
            prediction = sentiment_pipeline(sentence[:512])[0]  # Limiter à 512 tokens
            results.append({
                'phrase': sentence,
                'label': prediction['label'],
                'score': prediction['score']
            })

    if results:
        positive_count = sum(1 for r in results if r['label'] in ['5 stars', 'POSITIVE'])
        negative_count = sum(1 for r in results if r['label'] in ['1 stars', '2 stars', 'NEGATIVE'])
        # Déterminer le sentiment global
        if negative_count > positive_count:
            print("MÉCONTENTEMENT/FRUSTRATION")
        elif positive_count > negative_count:
            print("SATISFACTION")
        else:
            print("NEUTRE/MITIGÉ")
    return results


def main():
 
    file_path = Path("Je brise la glace.srt")
    
    if not file_path.exists():
        print(f"Erreur: Le fichier {file_path} n'existe pas!")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    customer_text = extract_customer_lines(text)
    
    # Analyser le sentiment
    results = analyze_sentiment_french(customer_text)


if __name__ == "__main__":
    main()
