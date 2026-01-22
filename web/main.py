import os

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from . import processor

# Obtenir le répertoire du script (web/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# Configuration
ALLOWED_EXTENSIONS = {"wav"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier reçu"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Aucun fichier sélectionné"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Lire les données audio directement depuis l'upload
        audio_data = file.read()

        # Récupérer le choix du premier locuteur
        first_speaker = request.form.get("first_speaker", "maif")

        # Call the external processor module
        try:
            analysis_result = processor.process_wav(
                audio_data, first_speaker=first_speaker
            )
        except Exception as e:
            return jsonify({"error": f"Erreur lors du traitement: {str(e)}"}), 500

        return (
            jsonify(
                {
                    "message": "Fichier téléversé avec succès",
                    "filename": filename,
                    "analysis": analysis_result,
                }
            ),
            200,
        )
    else:
        return (
            jsonify(
                {"error": "Type de fichier invalide. Seul le format WAV est autorisé."}
            ),
            400,
        )


if __name__ == "__main__":
    app.run(debug=True)
