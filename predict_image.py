import sys
import warnings
import numpy as np
import pickle

warnings.filterwarnings('ignore')

try:
    import cv2
    
    if len(sys.argv) < 2:
        print("ERROR: No image path provided.")
        sys.exit(1)
        
    img_path = sys.argv[1]
    
    # Load the trained model weights from Training.py (ANN)
    with open("weights_input_hidden.pickle", "rb") as f:
        weights_input_hidden = pickle.load(f)
    with open("weights_hidden_output.pickle", "rb") as f:
        weights_hidden_output = pickle.load(f)
    with open("bias_output.pickle", "rb") as f:
        bias_output = pickle.load(f)
    with open("bias_hidden.pickle", "rb") as f:
        bias_hidden = pickle.load(f)
        
    def relu(x):
        return np.maximum(0, x)

    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    IMG_SIZE = 150
    # Read image as grayscale, similar to training data
    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_array is None:
        print("ERROR: Failed to load image.")
        sys.exit(1)
        
    img_resize = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    
    # Preprocess image as done in Training/Loading_data
    # Convert image space to matching dimensions
    # Shape matching: Training uses X.shape=(n, 150, 150, 1) and X_flat=X.reshape(X.shape[0], -1) => (n, 22500)
    img_normalized = img_resize / 255.0
    img_flat = img_normalized.reshape(1, -1) # shape becomes (1, 22500)

    # Forward Pass through ANN
    hidden_layer_activation = np.dot(img_flat, weights_input_hidden) + bias_hidden
    hidden_layer_output = relu(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_activation)
    
    pred_val = predicted_output[0][0]
    
    print({
        "Normal": float(1.0 - pred_val),
        "Pneumonia": float(pred_val)
    })
    
        
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"ERROR: {str(e)}")
