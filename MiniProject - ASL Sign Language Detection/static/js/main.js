document.addEventListener('DOMContentLoaded', () => {
    
    // UI Elements
    const videoStream = document.getElementById('video-stream');
    const toggleCameraBtn = document.getElementById('toggle-camera');
    const clearWordBtn = document.getElementById('clear-word');
    const switchAlgoBtn = document.getElementById('switch-algo');
    const algoBadge = document.getElementById('algo-badge');
    
    const predictedLetterEl = document.getElementById('predicted-letter');
    const confidenceTextEl = document.getElementById('confidence-text');
    const confidenceFillEl = document.getElementById('confidence-fill');
    const formedWordEl = document.getElementById('formed-word');
    
    // Navigation Elements
    const navItems = document.querySelectorAll('.nav-item');
    const viewSections = document.querySelectorAll('.view-section');
    const viewTitle = document.getElementById('view-title');
    const viewSubtitle = document.getElementById('view-subtitle');
    const headerActions = document.querySelector('.header-actions');
    
    let cameraActive = true;
    let predictionInterval;
    let activeAlgorithm = 2; // Default

    // Navigation Logic
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = item.getAttribute('data-target');
            
            // Update Active Nav
            navItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            
            // Switch Views
            viewSections.forEach(section => {
                section.classList.remove('active');
                if (section.id === targetId) {
                    section.classList.add('active');
                }
            });

            // Update Header based on view
            updateHeader(targetId);
        });
    });

    function updateHeader(viewId) {
        if (viewId === 'live-section') {
            viewTitle.innerText = 'Real-Time ASL Translation';
            viewSubtitle.innerText = 'MediaPipe + SVM Powered Recognition';
            headerActions.style.display = 'flex';
        } else if (viewId === 'analytics-section') {
            viewTitle.innerText = 'Usage Analytics';
            viewSubtitle.innerText = 'Insights and recognition statistics';
            headerActions.style.display = 'none';
        } else if (viewId === 'settings-section') {
            viewTitle.innerText = 'System Settings';
            viewSubtitle.innerText = 'Configure your recognition experience';
            headerActions.style.display = 'none';
        }
    }

    // Toggle Camera
    toggleCameraBtn.addEventListener('click', () => {
        cameraActive = !cameraActive;
        if (cameraActive) {
            videoStream.style.display = 'block';
            toggleCameraBtn.innerHTML = '<i class="ph ph-video-camera-slash"></i> Stop Camera';
            startPredictionPolling();
        } else {
            videoStream.style.display = 'none';
            toggleCameraBtn.innerHTML = '<i class="ph ph-video-camera"></i> Start Camera';
            stopPredictionPolling();
            resetUI();
        }
    });

    // Clear Word
    clearWordBtn.addEventListener('click', async () => {
        try {
            await fetch('/api/clear_word', { method: 'POST' });
            formedWordEl.innerText = 'Waiting for input...';
            formedWordEl.style.opacity = '0.5';
        } catch (error) {
            console.error("Error clearing word:", error);
        }
    });

    // Switch Algorithm
    switchAlgoBtn.addEventListener('click', async () => {
        const nextAlgo = activeAlgorithm === 1 ? 2 : 1;
        
        try {
            const response = await fetch('/api/switch_algorithm', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ algorithm: nextAlgo })
            });
            const data = await response.json();
            
            if (data.success) {
                activeAlgorithm = data.active_algorithm;
                algoBadge.innerText = `Algorithm ${activeAlgorithm} Active`;
                
                // Visual feedback
                switchAlgoBtn.classList.add('btn-primary');
                setTimeout(() => switchAlgoBtn.classList.remove('btn-primary'), 500);
            }
        } catch (error) {
            console.error("Error switching algorithm:", error);
        }
    });

    // Poll Backend for Predictions
    function startPredictionPolling() {
        predictionInterval = setInterval(async () => {
            if (!cameraActive) return;
            
            try {
                const response = await fetch('/api/prediction');
                const data = await response.json();
                
                updateUI(data);
            } catch (error) {
                console.error("Error fetching prediction:", error);
            }
        }, 200); // 5 times per second
    }

    function stopPredictionPolling() {
        clearInterval(predictionInterval);
    }

    // Update UI Elements based on data
    function updateUI(data) {
        // Update Letter
        if (data.prediction && data.prediction !== 'None') {
            if (predictedLetterEl.innerText !== data.prediction) {
                predictedLetterEl.innerText = data.prediction;
                // Add pop animation
                predictedLetterEl.classList.add('updating');
                setTimeout(() => predictedLetterEl.classList.remove('updating'), 200);
            }
        } else {
            predictedLetterEl.innerText = '-';
        }

        // Update Confidence
        const confValue = parseFloat(data.confidence) * 100;
        confidenceTextEl.innerText = `${Math.round(confValue)}%`;
        confidenceFillEl.style.width = `${confValue}%`;
        
        // Color code confidence
        if (confValue > 80) {
            confidenceFillEl.style.background = 'linear-gradient(90deg, #10b981, #34d399)';
        } else if (confValue > 50) {
            confidenceFillEl.style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)';
        } else {
            confidenceFillEl.style.background = 'linear-gradient(90deg, #ef4444, #f87171)';
        }

        // Update Word
        if (data.word && data.word.trim() !== '') {
            formedWordEl.innerText = data.word;
            formedWordEl.style.opacity = '1';
        }
    }

    function resetUI() {
        predictedLetterEl.innerText = '-';
        confidenceTextEl.innerText = '0%';
        confidenceFillEl.style.width = '0%';
        confidenceFillEl.style.background = 'linear-gradient(90deg, var(--accent-primary), var(--accent-secondary))';
    }

    // Start polling on load
    startPredictionPolling();
});
