document.addEventListener('DOMContentLoaded', () => {
    const locator = document.getElementsByClassName('main-content')[0];

    // FILE INPUT FOR IMAGE UPLOAD
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'image/*';
    fileInput.style.display = 'none';
    locator.appendChild(fileInput);

    // UPLOAD BUTTON
    const button = document.createElement('button');
    button.textContent = 'Upload Image';
    locator.appendChild(button);

    button.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            // IMAGE DISPLAY LOGIC

            let img = document.querySelector('.dice-image');
            if (!img) {
                img = document.querySelector('.dice-image-alt');
            }
            if (img) {
                img.className = 'dice-image-alt';
                img.src = URL.createObjectURL(file);
            }
            
            // WHOLE DICE DETECTION LOGIC IS GOING TO BE HERE INSTEAD OF THE TIMEOUT
            // ---------------
            const detector = document.getElementsByClassName('detector')[0];
            detector.textContent = 'Detecting...';
            setTimeout(() => {
                const result = 'Detected: 6';
                detector.textContent = result;
            }, 2000);
            // ---------------
            
            /*
            const detector = document.getElementsByClassName('detector')[0];
            detector.textContent = 'Detecting...';

            const formData = new FormData();
            formData.append('file', file);

            fetch('http://localhost:5000/predict', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                detector.textContent = 'Detected: ' + data.prediction;
            })
            .catch(() => {
                detector.textContent = 'Detection failed!';
            });
            */
            }
    });
});