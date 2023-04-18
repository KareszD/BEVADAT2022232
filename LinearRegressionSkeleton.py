import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt


class LinearRegression:
    def __init__(self, epochs: int = 1000, lr: float = 1e-3):
        self.epochs = epochs
        self.lr = lr

    def fit(self, X: np.array, y: np.array):
        

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        # Building the model
        self.m = 0
        self.c = 0

        #L = 0.0001  # The learning Rate
        #epochs = 1000  # The number of iterations to perform gradient descent

        n = float(len(self.X_train)) # Number of elements in X

        # Performing Gradient Descent 
        losses = []
        for i in range(self.epochs): 
            y_pred = self.m*X_train + self.c  # The current predicted value of Y

            residuals = y_pred - y_train
            loss = np.sum(residuals ** 2)
            losses.append(loss)
            D_m = (-2/n) * sum(X_train * residuals)  # Derivative wrt m
            D_c = (-2/n) * sum(residuals)  # Derivative wrt c
            self.m = self.m + self.lr * D_m  # Update m
            self.c = self.c + self.lr * D_c  # Update c
            if i % 100 == 0:
                print(np.mean(y_train-y_pred))

    def predict(self, X):
        self.pred = []
        for X in self.X_test:
            y_pred = self.m*X + self.c
            self.pred.append(y_pred)
        print(self.pred)
        print(self.y_test)

        y_pred = self.m*self.X_test + self.c
        '''
        # Calculate the Mean Absolue Error
        print("Mean Absolute Error:", np.mean(np.abs(y_pred - self.y_test)))

        # Calculate the Mean Squared Error
        print("Mean Squared Error:", np.mean((y_pred - self.y_test)**2))

        plt.scatter(self.X_test, self.y_test)
        plt.plot([min(self.X_test), max(self.X_test)], [min(y_pred), max(y_pred)], color='red') # predicted
        plt.show()
        '''

'''
#print('GG')
#iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
#df
y = df['sepal length (cm)'].values
X = df['petal width (cm)'].values

LiRe = LinearRegression()
LiRe.__init__(lr=0.1)
LiRe.fit(X,y)
LiRe.predict(X)
'''