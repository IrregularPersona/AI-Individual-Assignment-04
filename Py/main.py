import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(hidden_size, input_size)
        self.biases_hidden = np.random.rand(hidden_size, 1)
        self.weights_hidden_output = np.random.rand(output_size, hidden_size)
        self.biases_output = np.random.rand(output_size, 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, input_data):
        # Hidden layer
        hidden = self.sigmoid(np.dot(self.weights_input_hidden, input_data) + self.biases_hidden)
        # Output layer
        output = self.sigmoid(np.dot(self.weights_hidden_output, hidden) + self.biases_output)
        return output, hidden

    def train(self, input_data, target, learning_rate):
        # Forward pass
        output, hidden = self.forward(input_data)

        # Compute output error and gradients
        output_error = output - target
        d_output = output_error * self.sigmoid_derivative(output)

        # Compute hidden layer error and gradients
        hidden_error = np.dot(self.weights_hidden_output.T, d_output)
        d_hidden = hidden_error * self.sigmoid_derivative(hidden)

        # Update weights and biases
        self.weights_hidden_output -= learning_rate * np.dot(d_output, hidden.T)
        self.biases_output -= learning_rate * d_output
        self.weights_input_hidden -= learning_rate * np.dot(d_hidden, input_data.T)
        self.biases_hidden -= learning_rate * d_hidden

def get_learning_rate(epoch):
    # Learning rate schedule
    if epoch < 200:
        return 0.01
    elif epoch < 500:
        return 0.005
    else:
        return 0.001

# Example usage with training data
if __name__ == "__main__":
    # Load and normalize data (dummy data here as an example)
    sales_data = np.loadtxt('inputfile.txt', delimiter=',')
    minval, maxval = np.min(sales_data), np.max(sales_data)
    normalized_data = 0.8 * ((sales_data - minval) / (maxval - minval)) + 0.1

    input_size = 12
    hidden_size = int((1.0 / 3.0) * input_size + 1)
    output_size = 1

    net = NeuralNetwork(input_size, hidden_size, output_size)

    num_epochs = 1000

    # Training
    for i in range(len(normalized_data) - input_size):
        input_data = normalized_data[i:i + input_size].reshape(-1, 1)
        target = np.array([[normalized_data[i + input_size]]])

        for epoch in range(num_epochs):
            learning_rate = get_learning_rate(epoch)
            net.train(input_data, target, learning_rate)

    # Prediction for next 6 months
    input_data = normalized_data[-input_size:].reshape(-1, 1)
    months_to_predict = 6

    predictions = []
    for month in range(months_to_predict):
        output, _ = net.forward(input_data)
        denormalized_prediction = ((output[0][0] - 0.1) / 0.8) * (maxval - minval) + minval
        predictions.append(denormalized_prediction)

        print(f"Predicted sales for month {month + 1}: {denormalized_prediction}")

        # Shift input data and add the latest prediction for future prediction
        input_data = np.roll(input_data, -1)
        input_data[-1] = (output[0][0] - 0.1) / 0.8
