# write your code here
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {'f1': [2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, 7.87],
        'f2': [65.2, 78.9, 61.1, 45.8, 54.3, 58.7, 96.1, 100.0, 85.9, 84.3],
        "f3": [25.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, 15.2],
        "y": [24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 27.1, 16.5, 18.9, 15.0]}

df = pd.DataFrame(data)


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = None
        self.intercept = None

    def fit(self, X, y):
        if self.fit_intercept:
            # Add a column of ones to the design matrix to allow fitting an intercept term
            X = np.hstack([np.ones((X.shape[0], 1)), X])
            # Solve the normal equations to find the optimal coefficients
        self.coefficient = np.linalg.solve(X.T.dot(X), X.T.dot(y))
        if self.fit_intercept:
            self.intercept = self.coefficient[0]
            self.coefficient = self.coefficient[1:]

        # else:
        #     self.coefficient = np.dot(np.linalg.inv(np.dot(X.T, X)), (np.dot(X.T, y)))

        # print(self.intercept)
        # print(self.coefficient)

    def predict(self, X):
        if self.fit_intercept:
            # Add a column of ones to the design matrix to allow fitting an intercept term
            X = np.hstack([np.ones((X.shape[0], 1)), X])
            # Make predictions using the optimal coefficients
        return X.dot(np.hstack([self.intercept, self.coefficient]))

    def r2_score(self, y, yhat):

        sse = np.sum((y - yhat)**2)
        sst = np.sum((y - np.mean(y))**2)
        return 1 - sse / sst

    def rmse(self, y, yhat):
        return np.sqrt(np.mean((y - yhat)**2))


regSci = LinearRegression(fit_intercept=True)

X = df[["f1", "f2", "f3"]]
y = df["y"]

regSci.fit(X, y)

y_hat = regSci.predict(X)


model = CustomLinearRegression(fit_intercept=True)
model.fit(X, y)
yhat = model.predict(X)

# Calculate the RMSE and R2 metrics
rmse = model.rmse(y, yhat)

r2 = model.r2_score(y, yhat)

results = {
    "Intercept": regSci.intercept_ - model.intercept,
    "Coefficient": regSci.coef_ - model.coefficient,
    "RMSE": np.sqrt(mean_squared_error(y, y_hat)) - rmse,
    "R2": r2_score(y, y_hat) - r2
}

print(results)
