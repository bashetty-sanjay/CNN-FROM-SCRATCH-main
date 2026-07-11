import numpy as np
import pickle
import gradio as gr
from sklearn.model_selection import train_test_split

# Load data
with open("X.pickle", "rb") as f:
    X = pickle.load(f)

with open("y.pickle", "rb") as f:
    y = pickle.load(f)

# Normalize data
X = X / 255.0

# Reshape y to be a column vector
y = y.reshape(-1, 1)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define Convolutional Layer
class Conv2D:
    def __init__(self, num_filters, filter_size, input_channels):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        self.filters = np.random.randn(num_filters, filter_size, filter_size, input_channels) * 0.1
    def forward(self, input):
        self.input = input
        input_height, input_width, input_channels = input.shape
        output_height = input_height - self.filter_size + 1
        output_width = input_width - self.filter_size + 1
        output = np.zeros((self.num_filters, output_height, output_width))
        for f in range(self.num_filters):
            for i in range(output_height):
                for j in range(output_width):
                    output[f, i, j] = np.sum(input[i:i+self.filter_size, j:j+self.filter_size, :] * self.filters[f])
        return output
    def backward(self, grad_output, learning_rate):
        grad_filters = np.zeros_like(self.filters)
        input_height, input_width, input_channels = self.input.shape

        for f in range(self.num_filters):
            for i in range(input_height - self.filter_size + 1):
                for j in range(input_width - self.filter_size + 1):
                    grad_filters[f] += (
                        self.input[i:i+self.filter_size, j:j+self.filter_size, :] * grad_output[f, i, j]
                    )

        # Update filters
        self.filters -= learning_rate * grad_filters

# Define Max Pooling Layer
class MaxPool2D:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, input):
        self.input = input
        num_filters, input_height, input_width = input.shape
        output_height = input_height // self.pool_size
        output_width = input_width // self.pool_size
        output = np.zeros((num_filters, output_height, output_width))

        for f in range(num_filters):
            for i in range(output_height):
                for j in range(output_width):
                    output[f, i, j] = np.max(
                        input[
                            f,
                            i*self.pool_size:i*self.pool_size+self.pool_size,
                            j*self.pool_size:j*self.pool_size+self.pool_size
                        ]
                    )
        return output

    def backward(self, grad_output):
        num_filters, input_height, input_width = self.input.shape
        grad_input = np.zeros_like(self.input)

        for f in range(num_filters):
            for i in range(grad_output.shape[1]):
                for j in range(grad_output.shape[2]):
                    window = self.input[
                        f,
                        i*self.pool_size:i*self.pool_size+self.pool_size,
                        j*self.pool_size:j*self.pool_size+self.pool_size
                    ]
                    mask = (window == np.max(window))
                    grad_input[
                        f,
                        i*self.pool_size:i*self.pool_size+self.pool_size,
                        j*self.pool_size:j*self.pool_size+self.pool_size
                    ] = mask * grad_output[f, i, j]
        return grad_input

# Define Fully Connected Layer
class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, grad_output, learning_rate):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)

        # Update weights and bias
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        return grad_input

# Define Dropout Layer
class Dropout:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, input, training=True):
        if training:
            self.mask = (np.random.rand(*input.shape) > self.dropout_rate).astype(float)
            return input * self.mask / (1 - self.dropout_rate)
        return input

    def backward(self, grad_output):
        return grad_output * self.mask / (1 - self.dropout_rate)

# Define the CNN
class CNN:
    def __init__(self, input_channels):
        self.conv1 = Conv2D(num_filters=8, filter_size=3, input_channels=input_channels)
        self.pool1 = MaxPool2D(pool_size=2)
        self.dropout1 = Dropout(dropout_rate=0.25)
        self.dense1 = Dense(input_size=8*74*74, output_size=64)
        self.dropout2 = Dropout(dropout_rate=0.5)
        self.dense2 = Dense(input_size=64, output_size=1)

    def forward(self, input, training=True):
        x = self.conv1.forward(input)
        x = relu(x)
        x = self.pool1.forward(x)
        x = self.dropout1.forward(x, training=training)
        x = x.reshape(1, -1)
        x = self.dense1.forward(x)
        x = relu(x)
        x = self.dropout2.forward(x, training=training)
        x = self.dense2.forward(x)
        x = sigmoid(x)
        return x

    def backward(self, grad_output, learning_rate):
        grad = grad_output
        grad = self.dense2.backward(grad, learning_rate)
        grad = relu_derivative(grad)
        grad = self.dropout2.backward(grad)
        grad = self.dense1.backward(grad, learning_rate)
        grad = grad.reshape(8, 74, 74)
        grad = self.dropout1.backward(grad)
        grad = self.pool1.backward(grad)
        grad = relu_derivative(grad)
        self.conv1.backward(grad, learning_rate)

#Binary Cross-Entropy Loss
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

input_channels = X.shape[3]

cnn = CNN(input_channels=input_channels)

# Training loop
learning_rate = 0.01
epochs = 5

for epoch in range(epochs):
    loss = 0
    for i in range(len(X_train)):
        # Forward pass
        output = cnn.forward(X_train[i], training=True)
        loss += binary_cross_entropy(y_train[i], output)

        # Backward pass
        grad_output = -(y_train[i] / output - (1 - y_train[i]) / (1 - output))
        cnn.backward(grad_output, learning_rate)

    print(f"Epoch {epoch+1}, Loss: {loss / len(X_train)}")

# Validation
correct = 0
for i in range(len(X_val)):
    output = cnn.forward(X_val[i], training=False)
    prediction = 1 if output > 0.5 else 0
    if prediction == y_val[i]:
        correct += 1

print(f"Validation Accuracy: {correct / len(X_val) * 100}%")

# Save the model (for Gradio/Predict.py)
def save_model(cnn):
    import pickle
    # Save in the format that Predict.py expects
    model_params = {
        "conv1_filters": cnn.conv1.filters,
        "dense1_weights": cnn.dense1.weights,
        "dense1_bias": cnn.dense1.bias,
        "dense2_weights": cnn.dense2.weights,
        "dense2_bias": cnn.dense2.bias
    }
    with open("model_params.pkl", "wb") as f:
        pickle.dump(model_params, f)
    # Also save the full model for compatibility
    with open("cnn_model.pkl", "wb") as f:
        pickle.dump(cnn, f)

save_model(cnn)

# Load the model
def load_model():
    import pickle
    with open("cnn_model.pkl", "rb") as f:
        return pickle.load(f)

# Gradio Interface
# def predict(image):
#     image = image / 255.0

#     cnn = load_model()

#     output = cnn.forward(image, training=False)
#     prediction = "Class 1" if output > 0.5 else "Class 0"
#     return prediction

