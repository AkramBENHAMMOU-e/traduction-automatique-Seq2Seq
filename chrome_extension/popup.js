document.addEventListener('DOMContentLoaded', () => {
    const translateBtn = document.getElementById('translateBtn');
    const sourceText = document.getElementById('sourceText');
    const targetText = document.getElementById('targetText');
    const status = document.getElementById('status');

    translateBtn.addEventListener('click', async () => {
        const text = sourceText.value.trim();
        if (!text) return;

        // UI State: Loading
        translateBtn.classList.add('loading');
        translateBtn.disabled = true;
        targetText.textContent = '';
        status.textContent = 'Translating...';

        try {
            const response = await fetch('http://localhost:5000/translate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();

            if (data.error) {
                targetText.textContent = 'Error: ' + data.error;
                status.textContent = 'Failed';
                status.style.color = 'red';
            } else {
                targetText.textContent = data.translation;
                status.textContent = 'Done';
                status.style.color = 'green';
            }
        } catch (error) {
            console.error(error);
            targetText.textContent = 'Connection Error. Is the API running?';
            status.textContent = 'Connection Failed';
            status.style.color = 'red';
        } finally {
            // UI State: Reset
            translateBtn.classList.remove('loading');
            translateBtn.disabled = false;
        }
    });

    // Optional: Enter key to translate
    sourceText.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            translateBtn.click();
        }
    });
});
