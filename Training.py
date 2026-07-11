import pickle
import numpy as np
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
# print(X.shape)
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
# print(X_flat.shape[1])
# He initialization: scale by sqrt(2 / n_inputs) — prevents always predicting one class
weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
weights_hidden_hidden = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
bias_hidden = np.zeros((1, hidden_size))
bias_output = np.zeros((1, output_size))
# print("wih:",np.dot(weights_input_hidden))
# print("who:",np.dot(weights_hidden_output))
# print("bh:",bias_hidden)
# print("bo:",bias_output)
epochs = 20
learning_rate = 0.001

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
        # update weights
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
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {float(np.squeeze(avg_loss).item()):.4f}, Accuracy: {accuracy:.2f}%')

# Save the model weights
with open("weights_input_hidden.pickle", "wb") as f:
    pickle.dump(weights_input_hidden, f)
with open("weights_hidden_output.pickle", "wb") as f:
    pickle.dump(weights_hidden_output, f)
with open("bias_output.pickle", "wb") as f:
    pickle.dump(bias_output, f)
with open("bias_hidden.pickle", "wb") as f:
    pickle.dump(bias_hidden, f)
