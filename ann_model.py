import numpy as np
import pickle

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Example data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Debug prints to verify the shape and content of X and y
print(f"X shape: {X.shape}")
print(f"X content: {X}")
print(f"y shape: {y.shape}")
print(f"y content: {y}")

# Flatten the input
if X.shape[0] == 0:
    raise ValueError("X array is empty. Please ensure it contains data.")
X_flat = X.reshape(X.shape[0], -1)

# Initialize weights and biases with a more robust initialization
input_size = X_flat.shape[1]
hidden_size = 16  # Increased hidden layer size
output_size = 1

weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
bias_output = np.zeros((1, output_size))

# Training parameters
epochs = 20000  # Increased number of epochs
learning_rate = 0.05  # Adjusted learning rate

# Training loop
for epoch in range(epochs):
    for i in range(X.shape[0]):
        # Forward pass
        hidden_layer_activation = np.dot(X_flat[i], weights_input_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        predicted_output = sigmoid(output_layer_activation)

        # Calculate error
        error = y[i] - predicted_output

        # Backpropagation
        d_predicted_output = error * sigmoid_derivative(predicted_output)
        error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # Update weights
        weights_hidden_output -= hidden_layer_output.reshape(hidden_size, 1).dot(d_predicted_output.reshape(1, output_size)) * learning_rate
        weights_input_hidden -= X_flat[i].reshape(input_size, 1).dot(d_hidden_layer.reshape(1, hidden_size)) * learning_rate

        # Update biases
        bias_output -= learning_rate * d_predicted_output
        bias_hidden -= learning_rate * np.sum(d_hidden_layer, axis=0, keepdims=True)
        
        # Debug prints
        if epoch % 5000 == 0 and i == 0:
            print(f"Epoch {epoch + 1}, Sample {i + 1}")
            print(f"Input: {X_flat[i]}")
            print(f"Hidden Layer Activation: {hidden_layer_activation}")
            print(f"Hidden Layer Output: {hidden_layer_output}")
            print(f"Output Layer Activation: {output_layer_activation}")
            print(f"Predicted Output: {predicted_output}")
            print(f"Error: {error}")
            print(f"Delta Predicted Output: {d_predicted_output}")
            print(f"Error Hidden Layer: {error_hidden_layer}")
            print(f"Delta Hidden Layer: {d_hidden_layer}")
            print(f"Updated Weights Hidden Output: {weights_hidden_output}")
            print(f"Updated Weights Input Hidden: {weights_input_hidden}")
            print(f"Updated Bias Output: {bias_output}")
            print(f"Updated Bias Hidden: {bias_hidden}")
            print("")

    # Print loss and accuracy every 5000 epochs
    if epoch % 5000 == 0:
        # Calculate loss
        loss = np.mean(np.abs(error))
        
        # Calculate accuracy
        predictions = np.round(predicted_output)
        correct_predictions = np.sum(predictions == y)
        accuracy = (correct_predictions / X.shape[0]) * 100
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}, Accuracy: {accuracy}')

# Save the model weights
with open("weights_input_hidden.pickle", "wb") as f:
    pickle.dump(weights_input_hidden, f)
with open("weights_hidden_output.pickle", "wb") as f:
    pickle.dump(weights_hidden_output, f)





import numpy as np
import os
import cv2
import pickle
import gradio as gr
training_data = []

IMG_SIZE = 150
datadir = r"D:\Codeeee\Project"
categories = ["NORMAL_TRAIN", "PNEUMONIA_TRAIN"]
def create_training_data():
    for category in categories:
        path = os.path.join(datadir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            # try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                # if img_array is None:
                #     print(f"Fail to load image {img}")
                #     continue
                img_resize = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([img_resize, class_num])
            # except Exception as e:
            #     print(f"Error in loading image {img}: {e}")
create_training_data()
# random.shuffle(training_data)
print(len(training_data))

x = []
y = []

for feature, label in training_data:
	x.append(feature)
	y.append(label)

X = np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = np.array(y)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open("Y.pickle","wb")
pickle.dump(Y, pickle_out)
pickle_out.close()


pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
pickle_in = open("Y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)
X_flat = X.reshape(X.shape[0], -1)

input_size = X_flat.shape[1] 
hidden_size = 64
output_size = 1

weights_input_hidden = np.random.randn(input_size, hidden_size)
weights_hidden_output = np.random.randn(hidden_size, output_size)
bias_hidden = np.zeros((1, hidden_size))
bias_output = np.zeros((1, output_size))
# print("wih:",np.dot(weights_input_hidden))
# print("who:",np.dot(weights_hidden_output))
# print("bh:",bias_hidden)
# print("bo:",bias_output)
epochs = 10
learning_rate = 0.01

for epoch in range(epochs):
    epoch_loss=0
    correct_predictions=0
    for i in range(X.shape[0]):
       
        hidden_layer_activation = np.dot(X_flat[i], weights_input_hidden)+bias_hidden
        hidden_layer_output = relu(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output)+bias_output
        predicted_output = sigmoid(output_layer_activation)

        # cal error
        loss = -(y[i] * np.log(predicted_output + 1e-8) + (1 - y[i]) * np.log(1 - predicted_output + 1e-8))
        epoch_loss += loss
        error = predicted_output - y[i]
        # bp
        d_predicted_output = error * sigmoid_derivative(predicted_output)
        error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * relu_derivative(hidden_layer_output)

        weights_hidden_output -= hidden_layer_output.reshape(hidden_size, 1).dot(d_predicted_output.reshape(1, output_size)) * learning_rate
        weights_input_hidden -= X_flat[i].reshape(input_size, 1).dot(d_hidden_layer.reshape(1, hidden_size)) * learning_rate

        bias_output -= learning_rate * d_predicted_output
        bias_hidden -= learning_rate * d_hidden_layer
        # print(predicted_output)
        # print(int(np.round(predicted_output)))
        if ((np.round(predicted_output)) == y[i]):
            correct_predictions += 1

    if epoch % 1 == 0:
        # print(correct_predictions)
        avg_loss=(epoch_loss/X.shape[0])
        accuracy = (correct_predictions / X.shape[0])*100
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {float(avg_loss)}%, Accuracy: {accuracy}%')

# Save the model weights
with open("weights_input_hidden.pickle", "wb") as f:
    pickle.dump(weights_input_hidden, f)
with open("weights_hidden_output.pickle", "wb") as f:
    pickle.dump(weights_hidden_output, f)



def predict(img):
    img = (np.array(img.convert("L").resize((IMG_SIZE, IMG_SIZE)))).flatten()/255

    # img = (img.reshape(IMG_SIZE, IMG_SIZE)).flatten() / 255.0
    # img = img.flatten() / 255.0 
    hidden_layer = relu(np.dot(img, weights_input_hidden) + bias_hidden)
    output = sigmoid(np.dot(hidden_layer, weights_hidden_output) + bias_output)
    print(output[0][0])
    return {"Normal": (1 - output[0][0]), "Pneumonia": (output[0][0])}

# test_img = X[5]
# test_label = y[5]
# prediction = predict(test_img)
# print(f"True Label: {test_label}, Prediction: {prediction}")


# iface = gr.Interface(
#     fn=predict,                     
#     inputs=gr.Image(type="pil"),     
#     outputs="text"              
# )
# iface.launch()
iface=gr.Interface(predict, gr.Image(), "text")
iface.launch()

# from PIL import Image

# Test image from dataset
# test_img = (X[5].reshape(IMG_SIZE, IMG_SIZE)).flatten() / 255.0  # Convert back to original image format
# test_img_pil = Image.fromarray(test_img.astype('uint8'))  # Convert to PIL image

# Call predict
# img = os.listdir(paath)
# imgarray = cv2.imread(os.path(r"D:\Codeeee\Project\new\img"), cv2.IMREAD_GRAYSCALE)
# imgresize = cv2.resize(imgarray, (150,150))
# prediction = predict(X[3])
# print("Prediction Output:", prediction)
