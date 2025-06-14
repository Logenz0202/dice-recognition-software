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

            const number_detector = document.getElementsByClassName('number_detector')[0];
            const shape_detector = document.getElementsByClassName('shape_detector')[0];
            let title_detector = document.getElementsByClassName('title_detector')[0];

            title_detector.textContent = `Title: ${file.name}`;
            number_detector.textContent = 'Detecting...';
            shape_detector.textContent = 'Detecting...';

            const formData = new FormData();
            formData.append('file', file);

            fetch('http://localhost:3000/detect', {
                method: 'POST',
                body: formData
            })
                .then(response => response.json())
                .then(data => {
                    if (data.result_Number && data.result_Shape) {
                        number_detector.textContent = `Detected: ${data.result_Number}`;
                        shape_detector.textContent = `Detected: ${data.result_Shape}`;
                        // Optionally update title again if backend returns a title
                        // title_detector.textContent = `Title: ${data.title || file.name}`;
                    } else {
                        number_detector.textContent = 'No dice detected';
                        shape_detector.textContent = 'No dice detected';
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    number_detector.textContent = 'Error detecting dice';
                    shape_detector.textContent = 'Error detecting dice';
                });
            // ...existing code...

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