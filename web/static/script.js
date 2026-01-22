const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const statusDiv = document.getElementById('status');
const fileNameDisplay = document.getElementById('fileName');
const resultsContent = document.getElementById('resultsContent');
const emptyState = document.getElementById('emptyState');

let selectedFile = null;

// Event Listeners for Drag & Drop
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

['dragenter', 'dragover'].forEach(eventName => {
    uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
});

uploadArea.addEventListener('drop', handleDrop, false);
uploadArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFiles);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles({ target: { files: files } });
}

function handleFiles(e) {
    const files = e.target.files;
    if (files.length > 0) {
        selectedFile = files[0];
        if (selectedFile.name.toLowerCase().endsWith('.wav')) {
            fileNameDisplay.textContent = selectedFile.name;
            uploadBtn.disabled = false;
        } else {
            fileNameDisplay.textContent = "Erreur : Seuls les fichiers .wav sont autorisés";
            uploadBtn.disabled = true;
            selectedFile = null;
        }
    }
}

uploadBtn.addEventListener('click', uploadFile);

async function uploadFile() {
    if (!selectedFile) return;

    // UI Updates
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<span class="loader"></span> Traitement en cours...';
    statusDiv.textContent = "";

    // Hide results if showing previous
    resultsContent.classList.add('hidden');
    emptyState.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            // Populate Results
            displayResults(data);
            statusDiv.innerHTML = "<span style='color:var(--maif-red)'>Analyse terminée</span>";
        } else {
            throw new Error(data.error || 'Échec du téléversement');
        }
    } catch (error) {
        statusDiv.textContent = `Erreur : ${error.message}`;
        statusDiv.style.color = 'red';
    } finally {
        uploadBtn.innerHTML = "Lancer l'analyse";
        uploadBtn.disabled = false;
    }
}

function displayResults(data) {
    emptyState.classList.add('hidden');
    resultsContent.classList.remove('hidden');

    // Populate fields
    if (data.analysis) {
        document.getElementById('transcriptText').textContent = `"${data.analysis.transcript}"`;
        document.getElementById('emotionPrimary').textContent = data.analysis.emotions.primary;
        document.getElementById('emotionConfidence').textContent = data.analysis.emotions.confidence + '%';
        document.getElementById('fileFilename').textContent = data.analysis.metadata.filename;
        document.getElementById('fileDuration').textContent = data.analysis.metadata.duration;
        document.getElementById('fileSampleRate').textContent = data.analysis.metadata.sample_rate;
    }
}

// Download JSON functionality
const downloadJsonBtn = document.getElementById('downloadJsonBtn');
let lastAnalysisData = null;

// Intercept displayResults to save data
const originalDisplayResults = displayResults;
displayResults = function (data) {
    lastAnalysisData = data;
    originalDisplayResults(data);
};

downloadJsonBtn.addEventListener('click', () => {
    if (!lastAnalysisData) return;

    const navName = lastAnalysisData.filename || 'analyse_audio';
    const jsonStr = JSON.stringify(lastAnalysisData, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `${navName}_resultats.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
});