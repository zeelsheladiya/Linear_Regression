#  write your code here 

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("/Users/zeelsheladiya/Documents/Zeel Data/Jetbrains Acedemy/Linear Regression/Topics/Linear "
                 "Regression in sklearn/Random regression/data/dataset/input.txt")
X_train = df.iloc[:-70,:4]
X_test = df.iloc[-70:, :4]
y_train = df.target[:-70]
y_test = df.target[-70:]

model = LinearRegression(fit_intercept=True)

model.fit(X_train, y_train)

print(round(model.intercept_, 3))