<h1 align="center">
  <img src="Logo_Maif_2019.svg.png" alt="MAIF" style="height: 200px;">
  <br />
</h1>

<p align="center">Prototype d’outil <b>local</b> d’analyse de conversations téléphoniques pour la MAIF, développé dans le cadre du <b>use case MAIF – AI4Industry</b>.
</p>

<p align="center"><i>"VocalisAI"</i></p>

---

L’application permet :

* le **prétraitement audio** (VAD),
* la **transcription avec diarisation**,
* la **synthèse structurée** des appels,
* via une **interface web Flask**.

---

## Prérequis

* **Python 3.12+** (voir `.python-version` / `pyproject.toml`)
* **pip** ou **uv**
* **GPU recommandé** (CUDA ou Apple Silicon MPS) pour WhisperX et les modèles LLM
* **Hugging Face token** (pour PyAnnote / diarisation WhisperX)

---

## Installation

### 1. Environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Installation des dépendances

Avec **pip** :

```bash
pip install -r requirements.txt
```

Avec **uv (recommandé)** :

```bash
uv sync
```

---

## Lancement de l’application

Dans un terminal à part, lancez Ollama:

```bash
ollama serve
```

Depuis la racine du projet :

```bash
uv run flask --app web/main.py run
```

L’interface est accessible sur :
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## Utilisation

1. Accéder à l’interface web
2. Charger un fichier audio (WAV, MP3, FLAC, etc.)
3. Lancer l’analyse
4. Les résultats affichés incluent :

   * transcription avec diarisation
   * résumé structuré MAIF
   * métadonnées audio

---

## Structure du projet (réelle)

```
USECASEMAIF/
├── .venv/                     # Environnement virtuel (local)
├── code_tests/                # Scripts et tests exploratoires
│
├── web/
│   ├── __init__.py
│   ├── main.py                # Point d’entrée Flask
│   ├── processor.py           # Pipeline principal (audio → texte → résumé)
│   ├── preprocessing.py       # Prétraitement audio (VAD rVADfast)
│   ├── summarize.py           # Synthèse structurée via LLM
│   ├── patch_lightning.py     # Monkey-patch PyTorch / Lightning (compatibilité WhisperX)
│   │
│   ├── templates/
│   │   └── index.html         # Interface utilisateur
│   │
│   └── static/
│       ├── script.js          # Logique frontend (upload, requêtes)
│       └── style.css          # Styles CSS
│
├── .env                       # Variables d’environnement (HF_TOKEN, etc.)
├── .gitignore
├── .python-version
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

---

## Description des modules clés

### `web/main.py`

* Initialise le serveur **Flask**
* Gère les routes HTTP
* Orchestration globale du pipeline

### `web/preprocessing.py`

* Prétraitement audio
* **Voice Activity Detection (rVADfast)**
* Normalisation, mono, resampling 16 kHz
* Fallback sécurisé si aucun speech détecté

### `web/processor.py`

* Pipeline principal :

  * appel du prétraitement
  * transcription WhisperX
  * gestion diarisation
  * collecte des résultats
* Extraction des métadonnées audio

### `web/summarize.py`

* Génération de **résumés structurés MAIF**
* Utilise `transformers` + modèle LLM local
* Format strict (problématique, résumé, actions, etc.)

### `web/patch_lightning.py`

* Monkey-patch de `lightning_fabric.utilities.cloud_io`
* Contournement du problème `torch.load(weights_only=True)`
* Nécessaire avec certaines versions de PyTorch / WhisperX

---

## Technologies utilisées

### Interface Web

* **Flask**
* HTML / CSS / JavaScript

### Audio & Transcription

* **WhisperX** : transcription + diarisation
* **PyTorch**
* **PyAnnote** (via WhisperX)
* **rVADfast** : Voice Activity Detection
* **audiofile**, **librosa**, **numpy**

### Résumé & NLP

* **transformers**
* **AutoModelForCausalLM**
* Modèles LLM locaux (GPU / MPS)

### Configuration

* **python-dotenv**
* **Pydantic**
* Variables d’environnement via `.env`

---

## Configuration importante

Créer un fichier `.env` à la racine :

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

---

## Notes importantes

* Les modèles sont téléchargés **au premier lancement**
* La diarisation nécessite un **token Hugging Face**
* Le traitement peut être long sur CPU
* Projet **prototype / expérimental**, non destiné à la production

---

## Statut du projet

* ✔ Pipeline fonctionnel
* ✔ Interface web opérationnelle
* ✔ Prétraitement audio robuste
* ⚠ Optimisations GPU / perfs en cours
* ⚠ Gestion des erreurs WhisperX dépendante des versions PyTorch
