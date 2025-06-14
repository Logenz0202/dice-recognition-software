const express = require('express');
const cors = require('cors');
const path = require('path');
const multer = require('multer');
const fs = require('fs');
const { execFile } = require('child_process');
const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir);
}
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads/');
    },
    filename: function (req, file, cb) {
        cb(null, Date.now() + '-' + file.originalname); // Keeps original name and extension
    }
});

const upload = multer({ storage: storage });

app.post('/detect', upload.single('file'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }
    const imagePath = path.resolve(req.file.path);

    execFile(
        'python',
        [
            'C:/Users/bstro/OneDrive/Dokumenty/STUDIA/sem4/Projects/dice-recognition-software/machine_learning/scripts/main.py',
            imagePath
        ],
        (error, stdout, stderr) => {
            fs.unlink(imagePath, () => { });

            const outputText = stdout.trim();
            if (!outputText) {
                // No output, check stderr
                console.error('Python script returned no output. Stderr:', stderr.trim());
                return res.status(500).json({
                    error: 'Python script did not return any output',
                    details: stderr.trim() || 'No details available'
                });
            }

            try {
                if (outputText.startsWith('{') || outputText.startsWith('[')) {
                    const output = JSON.parse(outputText);
                    const result_Number = output.face_value || 'Unknown';
                    const result_Shape = output.dice_type || 'Unknown';
                    return res.json({ result_Number, result_Shape });
                } else {
                    // Not JSON, treat as error
                    console.error('Python script did not return JSON:', outputText);
                    return res.status(500).json({
                        error: 'Python script did not return valid JSON',
                        details: outputText
                    });
                }
            } catch (e) {
                console.error('Error parsing output:', e);
                return res.status(500).json({ error: 'Invalid output from Python script' });
            }
        }
    );
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});