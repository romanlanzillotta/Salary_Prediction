import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("C:/Users/roman/PycharmProjects/Salary Prediction/Topics/Linear Regression in sklearn/Random regression/data/dataset/example.csv",sep=";")
print(df)
X = np.array(df["cost"]).reshape(-1, 1)
y = np.array(df["customers"]).reshape(-1, 1)
lr = LinearRegression().fit(X, y)
print(lr.predict(np.array((23)).reshape(1,-1)))
print(lr.coef_)