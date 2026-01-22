import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import processor

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'wavs'
ALLOWED_EXTENSIONS = {'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier reçu'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        
        # Call the external processor module
        try:
            analysis_result = processor.process_wav(save_path)
        except Exception as e:
            return jsonify({'error': f'Erreur lors du traitement: {str(e)}'}), 500
        
        return jsonify({
            'message': 'Fichier téléversé avec succès',
            'filename': filename,
            'analysis': analysis_result
        }), 200
    else:
        return jsonify({'error': 'Type de fichier invalide. Seul le format WAV est autorisé.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
