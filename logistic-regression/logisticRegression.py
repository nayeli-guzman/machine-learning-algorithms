import numpy as np

class LogisticRegression:

    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):

            # calculate y_pred = X*w + b
            linear = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear)

            # calculate gradients (derivatives|same as linearR)
            dw = 1/n_samples * np.dot(X.T, (y_pred - y))
            db = 1/n_samples * np.sum(y_pred - y)
    
            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear) # apply sigmoid to get probabilities
        y_pred = [1 if y >= 0.5 else 0 for y in y_pred] # convert probabilities to binary values
        return y_pred
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))