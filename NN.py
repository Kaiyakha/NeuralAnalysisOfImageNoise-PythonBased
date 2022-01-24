import numpy as np

sigmoid = lambda x: 1 / (1 + np.exp(-x * 1))
sigmoid_der = lambda x: sigmoid(x) * (1 - sigmoid(x))
wide_sigmoid = lambda x: 1 / (1 + np.exp(-x * 0.02))
wide_sigmoid_der = lambda x: wide_sigmoid(x) * (1 - wide_sigmoid(x))
ReLU = lambda x: np.maximum(0, x)
ReLU_der = lambda x: x > 0
linear = lambda x: x
linear_der = lambda x: 1


class NeuralNetwork:
    def __init__(self, shape: tuple):
        self.layers = len(shape)
        assert self.layers > 1
        self.shape = shape
        for l in shape: assert l > 0
        self.weights = [
            np.array([2 * np.random.random(shape[l - 1]) - 1 for _ in range(shape[l])])
            for l in range(1, self.layers)
        ]
        self.biases = [np.zeros(l) for l in shape[1:]]
        self.weighted_sums = [np.zeros(l) for l in shape[1:]]
        self.activation_functions = [wide_sigmoid] + [sigmoid for _ in range(self.layers - 2)]
        self.af_ders = [wide_sigmoid_der] + [sigmoid_der for _ in range(self.layers - 2)]
        self.activations = [np.zeros(l) for l in shape]
        self.deltas = [None for l in shape[1:]]

    def inspect(self):
        print("=============NeuralNetwork===============")
        print(f"Shape: {self.shape}")
        print(f"Weights: {self.weights}")
        print(f"Biases: {self.biases}")
        print(f"Activations: {self.activations}")

    def forward_prop(self, X):
        self.activations[0][:len(X)] = X[:self.shape[0]]
        for l in range(self.layers - 1):
            self.weighted_sums[l] = self.weights[l] @ self.activations[l] + self.biases[l]
            self.activations[l + 1] = self.activation_functions[l](self.weighted_sums[l])

    def backprop(self, Y, lr):
        self.deltas[-1] = (Y - self.activations[-1]) * self.af_ders[-1](self.weighted_sums[-1])
        for l in range(self.layers - 2, 0, -1):
            self.deltas[l - 1] = self.weights[l].T @ self.deltas[l] * self.af_ders[l - 1](self.weighted_sums[l - 1])
        for l in range(self.layers - 1):
            for j in range(self.shape[l + 1]):
                self.weights[l][j] += lr * self.activations[l] * self.deltas[l][j]
            self.biases[l] += self.deltas[l]

    def train(self, X, Y, lr, epochs, test_frequency = 1000):
        self.best_accuracy = 0
        for e in range(epochs):
            if not e % test_frequency:
                print(f"Epoch {e} | ", end = '')
                self.test(X, Y)
            i = np.random.randint(len(X))
            self.forward_prop(X[i])
            self.backprop(Y[i], lr)

    def test(self, X, Y):
        correct_predictions = predictions = 0
        for i in range(len(X)):
            self.forward_prop(X[i])
            ones = np.count_nonzero(Y[i])            
            if ones > 0: correct_predictions += len(np.intersect1d(np.argsort(self.activations[-1])[-ones:], np.argsort(Y[i])[-ones:]))
            predictions += ones
        accuracy = round(correct_predictions / predictions * 100, 1)
        if accuracy > self.best_accuracy: self.best_accuracy = accuracy
        print(f"Accuracy: {accuracy}% | Best accuracy: {self.best_accuracy}%", end = '\r')