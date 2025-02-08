import numpy as np

class SVM:
    def __init__(self, learning_rate=0.0001, n_iterations=300, _lambda=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self._lambda = _lambda
    
    def fit(self, X, y):

        # convert y {0, 1} to {-1, 1}
        y = np.where(y <= 0, -1, 1)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self._lambda * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self._lambda * self.weights - np.dot(x_i, y[idx]))
                    self.bias -= self.learning_rate * y[idx]

    def predict(self, X):
        
        linear = np.dot(X, self.weights) + self.bias
        return np.sign(linear)
    