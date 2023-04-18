import numpy as np



class LinearRegression:
    def __init__(self, epochs: int = 1000, lr: float = 1e-3):
        self.epoch = epochs
        self.learning_rate = lr
        self.s = 0
        self.y_intercept = 0

    def fit(self, X: np.array, y: np.array):
        n = float(len(X)) # Number of elements in X

        # Performing Gradient Descent
        losses = []
        for i in range(self.epoch):
            y_pred = self.s*X + self.y_intercept  # The current predicted value of Y

            residuals = y_pred - y
            loss = np.sum(residuals ** 2)
            losses.append(loss)
            D_s = (-2/n) * sum(X * residuals)  # Derivative wrt m
            D_yi = (-2/n) * sum(residuals)  # Derivative wrt c
            self.s = self.s + self.learning_rate * D_s  # Update m
            self.y_intercept = self.y_intercept + self.learning_rate * D_yi  # Update c
            # if i % 100 == 0:
                # print(np.mean(self.y_train-y_pred))

    def predict(self, X):
        y_pred = self.s* X + self.y_intercept
        return y_pred