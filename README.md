# MAIF AudioAnalyst - Interface Web

Prototype d'outil local d'analyse de conversations téléphoniques pour la MAIF, incluant transcription, diarisation et analyse émotionnelle, développé dans le cadre du useCase Maif - AI4Industry.

## Prérequis

- Python 3.12+ (requis par `pyproject.toml`)
- pip ou uv (gestionnaire de paquets)
- GPU (recommandé pour accélérer les modèles de transcription et d'analyse)

## Installation

### 1. Créer un environnement virtuel

**Linux / Mac :**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows :**
```bash
python -m venv venv
venv\Scripts\activate
```

Pour quitter l'environnement virtuel :
```bash
deactivate
```

### 2. Installer les dépendances Python

**Avec pip :**
```bash
pip install flask>=3.1.2 librosa>=0.11.0 pydantic>=2.12.5 pydantic-settings>=2.12.0 python-dotenv>=1.2.1 whisperx>=3.7.4 torch transformers audiofile numpy rVADfast pydub jiwer werkzeug
```

**Avec uv (recommandé) :**
```bash
uv sync
```

### 3. Structure des dossiers

##  Lancement

Depuis la racine du projet :

```bash
python web/website.py
```

L'application est accessible sur `http://127.0.0.1:5000`

## Utilisation

1. Accéder à l'interface web dans votre navigateur
2. Téléverser un fichier audio (WAV, MP3, FLAC, OGG, M4A, AAC, WMA) par glisser-déposer ou en cliquant sur la zone de dépôt
3. Sélectionner qui parle en premier (MAIF ou Sociétaire)
4. Cliquer sur "Lancer l'analyse"
5. Les résultats s'affichent dans le panneau de droite :
   - Transcription complète
   - Résumé structuré
   - Analyse émotionnelle
   - Métadonnées du fichier

## Technologies et Fonctionnalités

### Interface Web
- **Flask** (`flask>=3.1.2`) : Framework web pour créer l'API REST et servir l'interface utilisateur
- **Werkzeug** : Utilitaires Flask pour la gestion sécurisée des noms de fichiers (`secure_filename`)
- **HTML/CSS/JavaScript** : Interface utilisateur moderne avec design MAIF

### Transcription Audio
- **Whisper** (`openai-whisper`) : Modèle de transcription automatique de la parole (STT) développé par OpenAI
  - Support multilingue avec détection automatique de langue
  - Modèles disponibles : tiny, base, small, medium, large
  - Utilisé dans `train.py` et `Sentiment.py`
- **WhisperX** (`whisperx>=3.7.4`) : Extension de Whisper avec alignement de mots et diarisation intégrée
  - Améliore la précision de la transcription
  - Alignement temporel précis des mots
  - Utilisé pour la transcription avec diarisation dans le pipeline principal

### Diarisation (Identification des locuteurs)
- **WhisperX** : Intègre la diarisation pour identifier qui parle quand dans la conversation
  - Permet de distinguer les interventions de MAIF et du Sociétaire
  - Génère des segments temporels avec identification du locuteur
  - Format de sortie : SRT avec tags `[SPEAKER_XX]`

### Préprocessing Audio
- **rVADfast** (`rVADfast`) : Détection de voix active (Voice Activity Detection)
  - Identifie automatiquement les segments contenant de la parole
  - Filtre le silence et les bruits de fond
  - Utilisé dans `preprocessing.py` pour segmenter les fichiers audio
- **pydub** (`pydub`) : Manipulation et conversion de formats audio
  - Conversion entre formats (MP4, WAV, MP3, etc.)
  - Normalisation de la fréquence d'échantillonnage (16 kHz standard)
  - Conversion mono/stéréo
- **audiofile** (`audiofile`) : Lecture et écriture de fichiers audio
  - Interface simple pour charger/sauvegarder des signaux audio
  - Support de multiples formats
- **librosa** (`librosa>=0.11.0`) : Bibliothèque d'analyse audio avancée
  - Traitement du signal audio
  - Extraction de caractéristiques

### Analyse de Sentiment

Deux approches complémentaires ont été implémentées pour l'analyse de sentiment :

#### Approche 1 : Analyse avec LLM (Ollama)
- **Fichier** : `Sentiment.py`
- **Transcription** : **Whisper** pour convertir l'audio en texte
  - Modèles disponibles : tiny, base, small, medium, large
  - Support multilingue avec détection automatique
- **Analyse** : **Ollama** avec modèle LLM (par défaut `llama3`)
  - Analyse de satisfaction client via prompt structuré
  - Retourne un JSON avec : sentiment (satisfait/neutre/insatisfait), note (0-10), justification
  - Approche flexible et adaptable via prompts
  - Nécessite Ollama installé localement
- **Avantages** : Analyse contextuelle avancée, personnalisable via prompts
- **Cas d'usage** : Analyse globale de satisfaction client

#### Approche 2 : Analyse avec BERT après Diarisation
- **Fichier** : `sentiment_analysis_after_diarisation.py`
- **Transformers** (`transformers`) : Bibliothèque Hugging Face pour les modèles de NLP
- **Modèle BERT multilingue** (`nlptown/bert-base-multilingual-uncased-sentiment`) : 
  - Modèle pré-entraîné spécialisé dans l'analyse de sentiment
  - Optimisé pour le français et multilingue
  - Classification en 5 étoiles (1-5) ou labels (POSITIVE/NEGATIVE/NEUTRAL)
- **Fonctionnalités** :
  - Extraction automatique des lignes du client depuis la transcription SRT avec diarisation
  - Analyse phrase par phrase avec agrégation pour un sentiment global
  - Découpage intelligent du texte en phrases
  - Filtrage des phrases trop courtes (< 10 caractères)
  - Limitation à 512 tokens par phrase pour optimiser les performances
- **Avantages** : Rapide, spécialisé pour le sentiment, fonctionne après diarisation
- **Cas d'usage** : Analyse fine du sentiment du client uniquement (après séparation des locuteurs)

### Résumé et Synthèse
- **Transformers** avec **AutoModelForCausalLM** : Modèles de langage génératifs
- **Qwen/Qwen3-4B-Instruct-2507** : Modèle de langage instructif pour la génération de résumés
  - Produit des résumés structurés selon un format spécifique MAIF
  - Analyse la problématique principale, résumé global, et détails pour assureur/assuré
  - Utilisé dans `web/summarize.py`
  - Support de Flash Attention 2 pour l'optimisation GPU

### Évaluation de Performance
- **jiwer** (`jiwer`) : Bibliothèque pour calculer le Word Error Rate (WER)
  - Mesure la qualité de transcription en comparant avec la référence
  - Utilisé dans `evaluer_wer.py` pour l'évaluation des performances

### Traitement des Métadonnées Audio
- **wave** (bibliothèque standard Python) : Extraction des métadonnées des fichiers WAV
  - Fréquence d'échantillonnage
  - Durée du fichier
  - Nombre de canaux
  - Utilisé dans `web/processor.py` pour afficher les informations du fichier

### Infrastructure et Configuration
- **PyTorch** (`torch`) : Framework de deep learning
  - Support GPU (CUDA) pour accélération
  - Utilisé par tous les modèles de ML (Whisper, Transformers, etc.)
- **Pydantic** (`pydantic>=2.12.5`, `pydantic-settings>=2.12.0`) : Validation de données et gestion de configuration
- **python-dotenv** (`python-dotenv>=1.2.1`) : Gestion des variables d'environnement
- **NumPy** (`numpy`) : Calculs numériques et manipulation de tableaux
  - Utilisé pour le traitement des signaux audio

## Structure du Projet

```
UseCaseMaif/
├── web/
│   ├── website.py          # Serveur Flask principal
│   ├── processor.py        # Module de traitement audio et métadonnées
│   ├── summarize.py        # Module de résumé avec modèles de langage
│   ├── templates/
│   │   └── index.html      # Interface utilisateur
│   ├── static/
│   │   ├── script.js       # Logique client (upload, affichage résultats)
│   │   └── style.css       # Styles MAIF
│   └── wavs/              # Dossier uploads (créé auto)
├── src/
│   └── usecasemaif/       # Package principal avec pipeline de transcription
├── preprocessing.py        # Script de préprocessing avec VAD
├── sentiment_analysis_after_diarisation.py  # Analyse de sentiment post-diarisation
├── Sentiment.py           # Analyse de satisfaction avec Whisper + LLM
├── evaluer_wer.py         # Évaluation WER des transcriptions
├── train.py               # Script d'entraînement/test Whisper
├── pyproject.toml         # Configuration du projet et dépendances
└── README.md              # Ce fichier
```

## Configuration

- **Port** : Par défaut 5000 (modifiable dans `web/website.py`)
- **Mode debug** : Activé par défaut (`debug=True`)
- **Formats acceptés** : WAV, MP3, FLAC, OGG, M4A, AAC, WMA
- **Dossier uploads** : `web/wavs/`
- **Fréquence d'échantillonnage cible** : 16 kHz (standard pour la parole)
- **Mode audio** : Mono (recommandé pour la transcription)

## Développement

### Désactiver le mode debug en production

Modifier dans `web/website.py` :
```python
app.run(debug=False)
```

### Utilisation des modèles

- **Whisper** : Les modèles sont téléchargés automatiquement au premier usage
- **Transformers** : Les modèles sont téléchargés depuis Hugging Face au premier usage
- **GPU** : Détection automatique, utilise CUDA si disponible, sinon CPU

### Scripts utilitaires

- `preprocessing.py` : Prétraitement de fichiers audio avec VAD
- `evaluer_wer.py` : Évaluation de la qualité de transcription
- `train.py` : Test et validation des modèles Whisper

## Notes

- Les modèles de deep learning nécessitent une quantité significative de mémoire RAM/VRAM
- Pour de meilleures performances, utiliser un GPU avec CUDA
- Le traitement de fichiers longs peut prendre plusieurs minutes selon la configuration matérielle
