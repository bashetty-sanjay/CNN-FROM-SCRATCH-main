# Pediatric Pneumonia Detection via Custom-built CNN

This project provides an end-to-end Machine Learning pipeline and Web Application for detecting Pneumonia in pediatric chest X-rays.

What makes this project unique is that the **Convolutional Neural Network (CNN) is built entirely from scratch using pure NumPy** (`CNN_from_Scrach.py`), including manual implementations for:
- 2D Convolutions (`Conv2D`)
- Max Pooling Layers (`MaxPool2D`)
- Fully Connected Dense Layers (`Dense`)
- Dropout Layers (`Dropout`)
- Forward & Backward Propagation (Gradient Descent)

Alongside the from-scratch implementation, there are also validation models built using traditional Artificial Neural Networks (`Training.py` / `ann_model.py`) and TensorFlow/Keras sequential wrappers (`CNN.py`).

![UI Example Placeholder](https://via.placeholder.com/800x400?text=Pneumonia+Detection+Web+Application)

## Project Structure
- **`/web_app`**: Contains the full stack web application interface.
    - **`/frontend`**: A modern React application built with Vite providing a drag-and-drop UI.
    - **`server.js`**: An Express.js backend that handles image uploads and spawns the Python inference script.
- **`Loading_data.py`**: A dedicated script to preprocess the dataset, turn images into grayscale (150x150), and serialize the arrays via `pickle`.
- **`Training.py`**: Trains the standard Artificial Neural Network, saving the trained matrices for inference.
- **`CNN_from_Scrach.py`**: The pure-NumPy Convolutional Neural Network architecture.
- **`predict_image.py`**: The bridge script optimized for Node.js usage. Takes an image path from command-line arguments and outputs prediction JSON.

## Getting Started

### 1. Model Training & Data Prep
First, place your image datasets in a directory like `/Pediatric Chest X-ray Pneumonia/train` containing `NORMAL` and `PNEUMONIA` subdirectories.

1. **Serialize the Data**: Preprocess the dataset locally for faster disk loading.
   ```bash
   python Loading_data.py
   ```
2. **Train the Model**:
   ```bash
   python Training.py
   ```
   *(This will create the `weights_*.pickle` and `bias_*.pickle` needed for the web app.)*

### 2. Launching the Web Application
The web app is orchestrated via an Express server interacting with the Python bridge.

1. **Start the API Backend**:
   ```bash
   cd web_app
   npm install
   node server.js
   ```
2. **Start the React Frontend**:
   ```bash
   cd web_app/frontend
   npm install
   npm run dev
   ```

You can now visit your local server (e.g. `http://localhost:5174`) in the browser to interact with the trained Neural Network in real-time.
