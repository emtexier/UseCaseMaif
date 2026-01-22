# MAIF AudioAnalyst - Interface Web

Prototype d’outil local d’analyse de conversations téléphoniques pour la MAIF, incluant transcription, diarisation et analyse émotionnelle, développé dans le cadre du projet VocalisAI.

## 📋 Prérequis

- Python 3.11+
- pip

## 🚀 Installation

1. **Créer un environnement virtuel**

# Linux / Mac

python -m venv venv
source venv/bin/activate

# Windows

python -m venv venv
venv\Scripts\activate

# Pour quitter l’environnement virtuel :

```bash
deactivate
```

2. **Installer les dépendances Python**

```bash
pip install whisper torch subprocess transformers audiofile numpy rVADfast pydub jiwer processor werkzeug.utils
```

2. **Structure des dossiers**

Le dossier `wavs/` sera automatiquement créé au premier lancement pour stocker les fichiers uploadés.

## ▶️ Lancement

Depuis la racine du projet :

```bash
python web/website.py
```

L'application est accessible sur `http://127.0.0.1:5000`

## 🎯 Utilisation

1. Accéder à l'interface web dans votre navigateur
2. Téléverser un fichier audio (WAV) par glisser-déposer ou en cliquant sur la zone de dépôt
3. Sélectionner qui parle en premier (MAIF ou Sociétaire)
4. Cliquer sur "Lancer l'analyse"
5. Les résultats s'affichent dans le panneau de droite :
   - Transcription complète
   - Analyse émotionnelle
   - Métadonnées du fichier

## 📁 Structure

```
web/
├── website.py          # Serveur Flask principal
├── processor.py        # Module de traitement audio
├── templates/
│   └── index.html      # Interface utilisateur
├── static/
│   ├── script.js       # Logique client
│   └── style.css       # Styles MAIF
└── wavs/              # Dossier uploads (créé auto)
```

## ⚙️ Configuration

- **Port** : Par défaut 5000 (modifiable dans `website.py`)
- **Mode debug** : Activé par défaut (`debug=True`)
- **Formats acceptés** : WAV uniquement
- **Dossier uploads** : `web/wavs/`

## 🔧 Développement

Pour désactiver le mode debug en production, modifier dans `website.py` :

```python
app.run(debug=False)
```
