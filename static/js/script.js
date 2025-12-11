// Frontend logic for MRI upload and prediction

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const errorMessage = document.getElementById('errorMessage');
    const resultsCard = document.getElementById('resultsCard');
    const emptyState = document.getElementById('emptyState');

    let selectedFile = null;

    const resetUI = () => {
        resultsCard.style.display = 'none';
        emptyState.style.display = 'block';
        errorMessage.textContent = '';
        analyzeBtn.disabled = true;
        loadingSpinner.style.display = 'none';
    };

    const showError = (msg) => {
        errorMessage.textContent = msg;
    };

    const handleFiles = (files) => {
        if (!files || !files.length) return;
        selectedFile = files[0];
        analyzeBtn.disabled = false;
        errorMessage.textContent = `Selected: ${selectedFile.name}`;
    };

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

    clearBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        resetUI();
    });

    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) {
            showError('Please select an MRI image.');
            return;
        }

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('model_type', 'resnet');

        loadingSpinner.style.display = 'block';
        errorMessage.textContent = '';

        try {
            const resp = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await resp.json();
            loadingSpinner.style.display = 'none';

            if (!resp.ok) {
                showError(data.error || 'Prediction failed.');
                return;
            }

            renderResults(data);
        } catch (err) {
            loadingSpinner.style.display = 'none';
            showError('Network error. Try again.');
        }
    });

    const renderResults = (data) => {
        resultsCard.style.display = 'block';
        emptyState.style.display = 'none';

        // Prediction & confidence
        document.getElementById('predictedClass').textContent = data.predicted_class;
        document.getElementById('fileName').textContent = data.filename;
        document.getElementById('timestamp').textContent = new Date(data.timestamp).toLocaleString();

        const confidenceFill = document.getElementById('confidenceFill');
        const confidenceValue = document.getElementById('confidenceValue');
        const conf = Math.min(100, Math.max(0, data.confidence));
        confidenceFill.style.width = `${conf}%`;
        confidenceValue.textContent = `${conf.toFixed(1)}%`;

        // Probabilities
        const probBars = document.getElementById('probBars');
        probBars.innerHTML = '';
        const entries = Object.entries(data.probabilities || {});
        entries.forEach(([label, val]) => {
            const row = document.createElement('div');
            row.className = 'prob-row';
            row.innerHTML = `
                <div class="prob-label">${label}</div>
                <div class="prob-bar-wrap"><div class="prob-bar" style="width:${Math.min(100, val)}%"></div></div>
                <div class="prob-value">${val.toFixed(1)}%</div>
            `;
            probBars.appendChild(row);
        });

        // Images
        document.getElementById('originalImage').src = `data:image/png;base64,${data.input_image}`;
        document.getElementById('gradCamImage').src = `data:image/png;base64,${data.grad_cam}`;
    };

    // Mobile nav toggle
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    hamburger.addEventListener('click', () => {
        navMenu.style.display = navMenu.style.display === 'flex' ? 'none' : 'flex';
        navMenu.style.flexDirection = 'column';
        navMenu.style.gap = '12px';
    });

    resetUI();
});
