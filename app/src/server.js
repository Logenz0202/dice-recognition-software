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

    execFile('python', [path.join(__dirname, 'algo.py'), imagePath], (error, stdout, stderr) => {
        fs.unlink(imagePath, () => { });

        if (error) {
            console.error('Error running algo.py:', error);
            return res.status(500).json({ error: 'Internal server error' });
        }
        try {
            const output = JSON.parse(stdout.trim());
            const result_Number = output.face_value || 'Unknown';
            const result_Shape = output.dice_type || 'Unknown';
            res.json({ result_Number, result_Shape });
        } catch (e) {
            console.error('Error parsing output:', e);
            res.status(500).json({ error: 'Invalid output from Python script' });
        }
    });
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});