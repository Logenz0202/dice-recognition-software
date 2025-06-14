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

            const detector = document.getElementsByClassName('detector')[0];
            detector.textContent = 'Detecting...';

            // Simulate detection delay
            setTimeout(() => {
                fetch('http://localhost:3000/detect', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({}) // send empty object or relevant data
                })
                    .then(response => response.json())
                    .then(data => {
                        if (data.result) {
                            detector.textContent = `Detected: ${data.result}`;
                        } else {
                            detector.textContent = 'No dice detected';
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        detector.textContent = 'Error detecting dice';
                    });
            }, 2000);

            /*
            const formData = new FormData();
            formData.append('file', file);

            fetch('/detect', {
                method: 'POST',
                body: formData
            })
                .then(response => response.json())
                .then(data => {
                    if (data.result) {
                        detector.textContent = `Detected: ${data.result}`;
                    } else {
                        detector.textContent = 'No dice detected';
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    detector.textContent = 'Error detecting dice';
                });
            */
        }
    });
});