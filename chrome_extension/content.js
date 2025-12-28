let translateIcon = null;
let translatePopup = null;
let currentSelection = '';

// Create UI elements
function createUI() {
    // Floating Icon
    translateIcon = document.createElement('div');
    translateIcon.className = 'seq2seq-translate-icon';
    translateIcon.innerHTML = '🌐'; // Simple icon
    translateIcon.style.display = 'none';
    document.body.appendChild(translateIcon);

    // Translation Bubble
    translatePopup = document.createElement('div');
    translatePopup.className = 'seq2seq-translate-popup';
    translatePopup.style.display = 'none';
    document.body.appendChild(translatePopup);

    // Icon Click Event
    translateIcon.addEventListener('mousedown', (e) => {
        e.preventDefault();
        e.stopPropagation();
        translateSelection();
    });
}

function showIcon(x, y) {
    translateIcon.style.left = `${x}px`;
    translateIcon.style.top = `${y}px`;
    translateIcon.style.display = 'flex';
    translateIcon.classList.add('pop-in');
}

function hideUI() {
    if (translateIcon) translateIcon.style.display = 'none';
    if (translatePopup) translatePopup.style.display = 'none';
}

function translateSelection() {
    if (!currentSelection) return;

    // Show loading state in popup
    const rect = window.getSelection().getRangeAt(0).getBoundingClientRect();
    const x = rect.left + window.scrollX;
    const y = rect.bottom + window.scrollY + 10;

    translateIcon.style.display = 'none';

    translatePopup.innerHTML = '<div class="seq2seq-loader"></div> Translating...';
    translatePopup.style.left = `${x}px`;
    translatePopup.style.top = `${y}px`;
    translatePopup.style.display = 'block';

    chrome.runtime.sendMessage({ action: "translate", text: currentSelection }, (response) => {
        if (response && response.success) {
            if (response.data.error) {
                translatePopup.innerHTML = `<span style="color:#ef4444">Error: ${response.data.error}</span>`;
            } else {
                translatePopup.innerHTML = `
                    <div class="seq2seq-result">
                        <strong>French:</strong><br/>
                        ${response.data.translation}
                    </div>`;
            }
        } else {
            translatePopup.innerHTML = `<span style="color:#ef4444">Connection Failed</span>`;
        }
    });
}

document.addEventListener('mouseup', (e) => {
    // If clicking inside our popup, ignore
    if (translatePopup && translatePopup.contains(e.target)) return;
    if (translateIcon && translateIcon.contains(e.target)) return;

    const selection = window.getSelection().toString().trim();

    if (selection.length > 0) {
        currentSelection = selection;
        // Calculate position
        const rect = window.getSelection().getRangeAt(0).getBoundingClientRect();
        const x = rect.right + window.scrollX + 5;
        const y = rect.top + window.scrollY - 30;

        showIcon(x, y);
    } else {
        hideUI();
    }
});

// Initialize
createUI();
