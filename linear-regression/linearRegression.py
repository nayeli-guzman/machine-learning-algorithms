import numpy as np

class LinearRegression:

    def __init__(self, learning_rate=0.0001, n_iterations=300):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        # get number of samples and number of features

        n_samples, n_features = X.shape

        # initialize weights and bias to 0

        self.weights = np.zeros(n_features) # w for each feature
        self.bias = 0

        for _ in range(self.n_iterations):

            # calculate y_pred = X*w + b
            # X(s, f) * w(f, 1) = y_pred(s, 1)
            y_pred = np.dot(X, self.weights) + self.bias

            # calculate gradients (derivatives)
            # x.T(f, s) * (y_pred(s, 1) - y(s, 1)) = dw(f, 1)
            dw = 1/n_samples * np.dot(X.T, (y_pred - y))
            db = 1/n_samples * np.sum(y_pred - y)

            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

