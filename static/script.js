document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('detection-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = document.querySelector('.btn-text');
    const loader = document.querySelector('.loader');
    const scannerLine = document.querySelector('.scanner-line');
    const resultSection = document.getElementById('result-section');
    const resultBadge = document.getElementById('result-badge');
    const resultMessage = document.getElementById('result-message');
    const resultIcon = document.getElementById('result-icon');
    
    // Background orbs movement effect
    const orbs = document.querySelectorAll('.orb');
    document.addEventListener('mousemove', (e) => {
        const x = e.clientX / window.innerWidth;
        const y = e.clientY / window.innerHeight;
        
        orbs.forEach((orb, index) => {
            const speed = (index + 1) * 30;
            const xOffset = (x - 0.5) * speed;
            const yOffset = (y - 0.5) * speed;
            orb.style.transform = `translate(${xOffset}px, ${yOffset}px)`;
        });
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const headline = document.getElementById('headline').value.trim();
        const news = document.getElementById('news').value.trim();
        
        if (!headline && !news) {
            alert('Please provide either a headline or news content.');
            return;
        }

        // UI Loading State
        submitBtn.disabled = true;
        btnText.textContent = 'Analyzing...';
        loader.classList.remove('hidden');
        scannerLine.classList.remove('hidden');
        
        resultSection.classList.remove('show');
        setTimeout(() => resultSection.classList.add('hidden'), 500); // Wait for transition
        
        // Reset body class
        document.body.className = '';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ headline, news })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'An error occurred during analysis.');
            }

            // Small delay to make the scan feel more 'premium' even if API is fast
            await new Promise(resolve => setTimeout(resolve, 800));

            // Show Result
            resultBadge.textContent = data.prediction;
            resultBadge.className = 'badge';
            
            if (data.prediction === 'Fake') {
                resultBadge.classList.add('fake');
                resultMessage.textContent = 'Our AI has flagged this content as fabricated, manipulative, or highly misleading based on pattern analysis.';
                document.body.classList.add('theme-fake');
                resultIcon.className = 'ph ph-warning-octagon';
            } else {
                resultBadge.classList.add('real');
                resultMessage.textContent = 'Our AI analysis suggests this content is structurally sound and aligns with verifiable reporting patterns.';
                document.body.classList.add('theme-real');
                resultIcon.className = 'ph ph-shield-check';
            }
            
            resultSection.classList.remove('hidden');
            // Trigger reflow to restart animation
            void resultSection.offsetWidth;
            resultSection.classList.add('show');
            
            // Scroll to result slightly
            setTimeout(() => {
                resultSection.scrollIntoView({ behavior: 'smooth', block: 'end' });
            }, 100);

        } catch (error) {
            alert(error.message);
        } finally {
            // Restore UI
            submitBtn.disabled = false;
            btnText.textContent = 'Initiate Scan';
            loader.classList.add('hidden');
            scannerLine.classList.add('hidden');
        }
    });
});
