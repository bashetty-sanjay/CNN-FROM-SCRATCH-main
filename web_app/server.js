const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const port = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

// Setup Multer for handling file uploads
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        const uploadDir = path.join(__dirname, 'uploads');
        if (!fs.existsSync(uploadDir)) {
            fs.mkdirSync(uploadDir);
        }
        cb(null, uploadDir);
    },
    filename: function (req, file, cb) {
        cb(null, 'test-' + Date.now() + path.extname(file.originalname));
    }
});

const upload = multer({ storage: storage });

app.post('/api/predict', upload.single('image'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No image uploaded.' });
    }

    const imagePath = req.file.path;
    const scriptPath = path.join(__dirname, '..', 'predict_image.py');

    // Need to execute from the Python project directory so relative paths in Python work
    const cwdPath = path.join(__dirname, '..');

    const pythonCommand = `python "${scriptPath}" "${imagePath}"`;

    exec(pythonCommand, { cwd: cwdPath }, (error, stdout, stderr) => {
        // Clean up the uploaded file after processing
        fs.unlink(imagePath, (err) => {
            if (err) console.error("Failed to delete temp image:", err);
        });

        if (error) {
            console.error(`Python script execution error: ${error.message}`);
            if (!res.headersSent) {
                return res.status(500).json({ error: 'Failed to process image', details: stderr || error.message });
            }
            return;
        }

        try {
            const resultString = stdout.trim();
            let predictionResult = resultString;

            // Output looks like "{'Normal': 0.99, 'Pneumonia': 0.01}" or "Normal"
            if (resultString.includes("{") && resultString.includes("}")) {
                // Basic parsing if it returned a dict string
                const jsonStr = resultString.replace(/'/g, '"');
                try {
                    predictionResult = JSON.parse(jsonStr);
                } catch (e) {
                    // Fallback if parsing fails
                    predictionResult = resultString;
                }
            }
            res.json({ prediction: predictionResult });
        } catch (err) {
            res.status(500).json({ error: 'Failed to parse python output.', details: resultData });
        }
    });
});

app.listen(port, () => {
    console.log(`Pneumonia Prediction API listening at http://localhost:${port}`);
});
