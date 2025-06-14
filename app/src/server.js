const express = require('express');
const cors = require('cors');
const path = require('path');
const app = express();
const PORT = process.env.PORT || 3000;

// Enable CORS for all routes
app.use(cors());

// Middleware to parse JSON bodies
app.use(express.json());

// Sample route
app.get('/', (req, res) => {
    res.json({ message: 'Hello, World!' });
});

// Dummy /detect endpoint for testing (LATER CHANGED TO REAL DETECTION LOGIC)
app.post('/detect', (req, res) => {
    // Return a random dice result between 1 and 6
    const randomResult = Math.floor(Math.random() * 6) + 1;
    res.json({ result: randomResult });
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});