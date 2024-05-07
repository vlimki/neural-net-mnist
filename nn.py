import numpy as np

X = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])
Y = np.array([0, 1, 1, 0])

n_l1 = 4
n_l2 = 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse(Y: np.ndarray, Yhat: list[int]):
    return 1 / (2 * len(Yhat)) * sum([(Y[i] - Yhat[i]) ** 2 for i in range(len(Yhat))])


class Neuron:
    def __init__(self, activation):
        self.activation = activation

    def activate(self, X, W):
        return sigmoid(np.matmul(W, X))

class Layer:
    def __init__(self, n):
        self.n = n
        self.neurons = []

        for _ in range(n):
            self.neurons.append(Neuron(activation=sigmoid))

    def calculate(self, X, W) -> np.ndarray:
        vec = np.array([])

        for i in range(len(self.neurons)):
            vec = np.append(vec, self.neurons[i].activate(X=X, W=W[i, :]))

        return vec

class Network:
    def __init__(self, layers):
        self.layers = layers
        # here
        self.weights = []

    def fit(self, X):
        input_next = X
        for i in range(len(self.layers)):
            rows = self.layers[i].n
            cols = input_next.shape[0]

            W = np.random.rand(cols, rows)
            self.weights.append(W)
            input_next = self.layers[i].calculate(input_next, self.weights[i].T)

    def predict(self, X):
        input_next = X

        for i in range(len(self.layers)):
            input_next = self.layers[i].calculate(input_next, self.weights[i].T)

        return input_next

    def gradient_descent(self, X, Y):
        results = []
        for i in range(len(X)):
            res = a.predict(X[i])
            results.append(res)

        cost = mse(Y, results)

        print(f"ERROR: {mse(Y, results)}")

        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - 0.005 * ddx
        
        return None

    def train(self, X, Y, epochs):
        for _ in range(epochs):
            self.gradient_descent(X, Y)

        return None
        
a = Network(layers=[
    Layer(n=4),
    Layer(n=1)
])

a.fit(np.array([1, 0]))
a.train(X, Y, epochs=10000)

print(a.predict([0, 0]))
print(a.predict([1, 0]))
print(a.predict([0, 1]))
print(a.predict([1, 1]))

for i in range(0):
    results = []

    for i in range(len(X)):
        res = a.predict(X[i])
        y = Y[i]
        print(f"Input: {X[i]} | Target: {Y[i]}, Prediction: {res}")
        results.append(res)

    print(sigmoid_derivative(mse(Y, results)))
